"""
Bittle X hardware adapter.

Translates ReactionAction objects (produced by the VA pipeline) into serial
commands for the Petoi Bittle X ESP32 over USB/Bluetooth.

Integration points
------------------
run_offline.py  — passes adapter to run_offline_session(); called after each
                  stability-gated behavior dispatch.
run_online.py   — passed as on_behavior_update callback to StreamingSession;
                  called by _dispatch_behavior() immediately when each analysis
                  worker reaches a decision.

Graceful degradation
--------------------
If pyserial is not installed, or the specified port cannot be opened, the
adapter automatically falls back to mock mode: every command is printed to
the terminal prefixed with [BITTLE-MOCK] but no exception is raised.  The rest
of the pipeline continues unaffected.
"""

import time
import threading
import logging

try:
    import serial
except ImportError:
    serial = None
    logging.warning(
        "pyserial not installed — BittleXAdapter will run in mock mode. "
        "Install with: pip install pyserial"
    )

# Minimum seconds between distinct intent executions.  Prevents servo jitter
# caused by rapid intent switches when the VA pipeline emits decisions faster
# than the robot can mechanically settle between poses.
INTENT_COOLDOWN_SEC = 2.5

# Servo transition speed sent before every skill command.
# Range: 0 (slowest) to 10 (fastest); firmware default is ~8.
# NOTE: Petoi firmware versions differ on the command format.
#   Most recent firmware:  "s 4"  (with space, used here)
#   Older firmware:        "s4"   (no space)
# If speed control has no visible effect, remove the space and retry.
SERVO_SPEED = 4


class BittleXAdapter:
    """
    Serial adapter for the Petoi Bittle X robot dog.

    Accepts ReactionAction objects from the VA pipeline and maps their intent
    field to the appropriate Bittle skill command over the serial port.

    Intent → command mapping
    ------------------------
    NEUTRAL      → kbalance  (standing balance)
    CHECK_IN     → knd       (nod)
    ENGAGE       → khi       (wave / greet)
    DE_ESCALATE  → kbk then ksit  (back up 1.5 s, then sit)
    CAUTION      → kbalance  (rigid stance)
    """

    # Maps pipeline intents to Bittle skill names.
    # Tuple layout: (command, motion budget in seconds).
    # The budget is the realistic physical duration of each motion — it controls
    # how long _executing stays True, which determines when the queue releases
    # the next pending intent.  Tune against observed hardware behaviour.
    _INTENT_COMMANDS = {
        "NEUTRAL":     [("kbalance", 1.5)],
        "CAUTION":     [("kbalance", 1.5)],
        "CHECK_IN":    [("knd",      2.0)],
        "ENGAGE":      [("khi",      3.5)],   # full wave needs time to complete
        "DE_ESCALATE": [("kbk",      2.0), ("ksit", 1.5)],  # back up + sit
    }

    def __init__(self, port: str = "COM3", baudrate: int = 115200, mock: bool = False):
        self.port      = port
        self.baudrate  = baudrate
        self.mock      = mock or (serial is None)
        self.serial_conn = None
        self.current_pose: str | None = None

        # Walking timer — Bittle walks run indefinitely until stopped
        self.is_walking    = False
        self.walking_timer: threading.Timer | None = None

        # Cooldown tracking — wall-clock time of the last executed intent
        self._last_intent_time: float = 0.0

        # Single-slot reaction queue — absorbs pipeline bursts without backlog
        self._pending_intent: str | None = None  # the waiting slot
        self._executing: bool = False             # True while a motion is running

        self._connect()

    # ── Connection ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self.mock:
            print(f"[BITTLE-MOCK] Mock mode active — no serial connection to {self.port}.")
            return

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for ESP32 to reset after DTR toggle
            print(f"[BITTLE] Connected on {self.port} @ {self.baudrate} baud.")
            # Wake up and balance
            self.send_command("kbalance")
            time.sleep(0.5)
        except Exception as exc:
            msg = str(exc)
            print(f"\n[BITTLE] Connection failed on {self.port}: {msg}")
            if "Access is denied" in msg or "PermissionError" in msg:
                print("[BITTLE] Tip: close Arduino IDE or any other app using the port.")
            print("[BITTLE] Falling back to mock mode for this session.")
            self.mock = True

    def disconnect(self) -> None:
        """Send rest command and close serial port."""
        self._stop_walking()
        self.send_command("d")  # Rest/sleep — relaxes servos and saves battery
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        print("[BITTLE] Disconnected.")

    # ── Core serial I/O ────────────────────────────────────────────────────────

    def _send_smooth(self, cmd: str) -> None:
        """
        Send a servo speed-control prefix then a skill command.

        Use this for all expressive motion commands (kbalance, knd, khi, kbk,
        ksit, etc.) so servo transitions are gradual.  Do NOT use for utility
        or safety commands (p, d, kbalance in _connect) — those should bypass
        speed control so they respond immediately.
        """
        self.send_command(f"s {SERVO_SPEED}")
        time.sleep(0.05)   # small gap so firmware processes speed before skill
        self.send_command(cmd)

    def send_command(self, cmd: str) -> None:
        """Write a single skill or control command to the Bittle serial port."""
        if self.mock:
            print(f"[BITTLE-MOCK] → {cmd}")
            return

        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{cmd}\n".encode("utf-8"))
                print(f"[BITTLE] → {cmd}")
            except Exception as exc:
                print(f"[BITTLE] Send error for '{cmd}': {exc}")

    # ── Pipeline integration ───────────────────────────────────────────────────

    def apply_reaction(self, reaction_action) -> None:
        """
        Primary integration point — called by both run_offline.py and
        run_online.py after each stability-gated behavior dispatch.

        Accepts a ReactionAction dataclass (pipeline.reaction_action) and
        executes the mapped Bittle skill sequence for its .intent field.
        """
        print(f"[BITTLE-DEBUG] apply_reaction called, "
              f"intent={getattr(reaction_action, 'intent', None)}, "
              f"ra={reaction_action}")
        if reaction_action is None:
            return
        intent = getattr(reaction_action, "intent", None)
        if intent:
            self._dispatch_or_queue(intent)

    # Alias kept for backwards-compatibility with any older callers
    apply_reaction_pose = apply_reaction

    # ── Reaction queue ─────────────────────────────────────────────────────────

    def _dispatch_or_queue(self, intent: str) -> None:
        """
        Single-slot queue layer — sits between apply_reaction and _execute_intent.

        If the robot is free, execute immediately.
        If busy, hold the intent in the single pending slot.  If the slot is
        already occupied, the incoming intent replaces the stale one — the robot
        always picks up the most recent emotional content rather than replaying
        an outdated backlog.
        """
        if not self._executing:
            self._run_intent(intent)
        else:
            if self._pending_intent is not None and self._pending_intent != intent:
                print(f"[BITTLE] Replacing stale pending '{self._pending_intent}' "
                      f"with '{intent}'")
            self._pending_intent = intent

    def _run_intent(self, intent: str) -> None:
        """
        Execute intent in a background thread; pick up pending on finish.

        Sets _executing=True for the full physical duration of the motion
        (controlled by the motion budget in _INTENT_COMMANDS), then clears
        it and dispatches any pending intent that arrived while busy.
        The thread is daemonised so it never blocks pipeline shutdown.
        """
        self._executing = True

        def _worker():
            try:
                self._execute_intent(intent)
            finally:
                self._executing = False
                # Consume the slot atomically before calling back in
                next_intent = self._pending_intent
                self._pending_intent = None
                if next_intent is not None:
                    print(f"[BITTLE] Picking up pending intent: {next_intent}")
                    self._run_intent(next_intent)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── Intent execution ───────────────────────────────────────────────────────

    def _execute_intent(self, intent: str) -> None:
        """
        Map a pipeline intent string to one or more serial commands.

        Guards against two sources of servo jitter:

        1. Deduplication — if the robot is already executing this intent the
           call is a no-op, preventing redundant serial writes for sustained
           states.

        2. Rate limiting (INTENT_COOLDOWN_SEC) — a wall-clock cooldown rejects
           any new intent that arrives sooner than INTENT_COOLDOWN_SEC after
           the previous one.  This absorbs rapid VA pipeline decisions that
           would otherwise thrash the servos mid-motion.

        3. Settle delay — single-step intents sleep for the delay stored in
           _INTENT_COMMANDS after sending the command, giving servos time to
           complete the motion before the caller can dispatch the next intent.
        """
        intent = intent.upper()

        if intent == self.current_pose:
            return  # Already in this state; skip redundant command

        now = time.time()
        if now - self._last_intent_time < INTENT_COOLDOWN_SEC:
            print(f"[BITTLE] Cooldown active — skipping intent '{intent}'")
            return
        self._last_intent_time = now

        self.current_pose = intent
        self._stop_walking()

        commands = self._INTENT_COMMANDS.get(intent)
        if commands is None:
            print(f"[BITTLE] Unknown intent '{intent}' — defaulting to kbalance.")
            self._send_smooth("kbalance")
            return

        print(f"[BITTLE] Executing intent: {intent}")

        if len(commands) == 1:
            self._send_smooth(commands[0][0])
            settle = commands[0][1]
            if settle > 0:
                time.sleep(settle)  # Let servos finish before next command arrives
        else:
            # Multi-step sequence: send first command then schedule the rest
            self._send_smooth(commands[0][0])
            delay = commands[0][1]
            if delay > 0:
                self._schedule_sequence(commands[1:], delay)

    def _schedule_sequence(self, remaining_commands, delay: float) -> None:
        """Fire remaining commands in sequence after a delay."""
        def _fire():
            for cmd, next_delay in remaining_commands:
                self._send_smooth(cmd)
                if next_delay > 0:
                    time.sleep(next_delay)
            self.is_walking = False

        self.is_walking = True
        t = threading.Timer(delay, _fire)
        t.daemon = True
        t.start()
        self.walking_timer = t

    # ── Walking / continuous motion helpers ────────────────────────────────────

    def _start_walking_timer(self, duration: float = 2.0, stop_cmd: str = "p") -> None:
        """Bittle walks run indefinitely — schedule an automatic stop."""
        self._stop_walking()
        self.is_walking = True

        def _stop():
            if self.is_walking:
                self.send_command(stop_cmd)
                self.is_walking = False

        self.walking_timer = threading.Timer(duration, _stop)
        self.walking_timer.daemon = True
        self.walking_timer.start()

    def _stop_walking(self) -> None:
        self.is_walking = False
        if self.walking_timer:
            self.walking_timer.cancel()
            self.walking_timer = None

    # ── Convenience pose shortcuts ─────────────────────────────────────────────

    def stand(self) -> None:
        self.send_command("kstr")
        self.current_pose = "STAND"
        self._stop_walking()

    def sit(self) -> None:
        self.send_command("ksit")
        self.current_pose = "SIT"
        self._stop_walking()

    def stop(self) -> None:
        self.send_command("p")
        self._stop_walking()
