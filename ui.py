# ui.py
import tkinter as tk
from tkinter import scrolledtext
from typing import Callable, Optional

class AppUI:
    """All Tkinter widgets plus thread-safe text writes via ``root.after``."""

    def __init__(
        self,
        root: tk.Tk,
        on_user_input: Callable[[str], None],
        on_reingest: Callable[[], None],
    ) -> None:
        self.root = root
        self._on_user_input = on_user_input
        self._on_reingest = on_reingest

        self.root.title("Live Interview Translator")
        self.root.geometry("900x600")

        self.chat_display = scrolledtext.ScrolledText(
            root, font=("Helvetica", 16), wrap=tk.WORD, state="disabled"
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(root)
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.input_entry = tk.Entry(input_frame, font=("Helvetica", 16))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", self._handle_input)
        self.input_entry.focus()

        send_btn = tk.Button(
            input_frame,
            text="Send / Translate",
            font=("Helvetica", 14),
            command=self._handle_input,
        )
        send_btn.pack(side=tk.RIGHT, padx=(0, 10))

        reingest_btn = tk.Button(
            input_frame,
            text="🔄 Re-ingest PDFs",
            font=("Helvetica", 14),
            command=self._handle_reingest,
        )
        reingest_btn.pack(side=tk.RIGHT)

    def write(self, text: str) -> None:
        """Thread-safe: schedule the actual widget update on the main thread."""
        self.root.after(0, self._safe_write, text)

    def _safe_write(self, text: str) -> None:
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def _handle_input(self, _event: Optional[tk.Event] = None) -> None:
        user_text = self.input_entry.get().strip()
        if user_text:
            self._on_user_input(user_text)
            self.input_entry.delete(0, tk.END)

    def _handle_reingest(self) -> None:
        self._on_reingest()