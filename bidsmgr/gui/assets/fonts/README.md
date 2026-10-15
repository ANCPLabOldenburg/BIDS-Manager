# Bundled typefaces

BIDS Manager draws every piece of its interface in the same two typefaces on
macOS, Windows and Linux, so it looks the same everywhere and no text falls
back to a font a system happens to have.

| Typeface | Files | Version | Source | Licence |
|---|---|---|---|---|
| Inter | `Inter-Regular.ttf`, `Inter-Medium.ttf`, `Inter-SemiBold.ttf`, `Inter-Bold.ttf` | 4.1 (static TTF from `extras/ttf`) | https://github.com/rsms/inter/releases/tag/v4.1 | SIL Open Font License 1.1, `Inter-OFL.txt` |
| JetBrains Mono | `JetBrainsMono-Regular.ttf`, `JetBrainsMono-Bold.ttf` | 2.304 | https://github.com/JetBrains/JetBrainsMono/releases/tag/v2.304 | SIL Open Font License 1.1, `JetBrainsMono-OFL.txt` |

Inter is the interface (`bidsmgr.gui.typefaces.UI_FAMILY`); JetBrains Mono is
for code, logs and raw text only (`MONO_FAMILY`). Both are registered with Qt
by `bidsmgr.gui.typefaces.load()` before the stylesheet is applied. The files
are unmodified; the licence permits bundling them with software, and each
licence file travels with its fonts.
