"""Command-line launcher for the Cobasket portfolio editor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from .config_editor import ConfigEditorDialog


def build_parser() -> argparse.ArgumentParser:
    """Build the portfolio-editor argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting an optional portfolio JSON path.
    """
    parser = argparse.ArgumentParser(description="Edit a Cobasket portfolio and watchlist")
    parser.add_argument("config", nargs="?", help="portfolio configuration JSON")
    return parser


def main() -> None:
    """Launch the portfolio and watchlist editor."""
    args = build_parser().parse_args()
    app = QApplication(sys.argv)
    config_path = args.config
    if config_path is None:
        config_path, _ = QFileDialog.getOpenFileName(
            None,
            "Choose portfolio configuration",
            str(Path.cwd()),
            "JSON files (*.json);;All files (*)",
        )
    if not config_path:
        return
    if not Path(config_path).exists():
        QMessageBox.critical(None, "Missing file", "The selected portfolio file does not exist.")
        return
    dialog = ConfigEditorDialog(config_path)
    dialog.exec()


if __name__ == "__main__":
    main()
