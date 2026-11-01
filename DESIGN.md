---
name: SpeechDock
description: A calm, local speech workspace built with Lucent 2 and Rosé Pine.
colors:
  dark-canvas: "#191724"
  dark-sheet: "#1F1D2EE8"
  dark-glass: "#120E1A66"
  dark-edge: "#FFFFFF1F"
  dark-ink: "#F2EFF6"
  dark-ink-muted: "#F2EFF6A8"
  dark-accent: "#A79BF5"
  dark-accent-ink: "#17131F"
  dark-positive: "#9CCFD8"
  dark-danger: "#EB6F92"
  light-canvas: "#FAF4ED"
  light-sheet: "#FFFAF3EB"
  light-glass: "#FFFFFF85"
  light-edge: "#FFFFFFD9"
  light-ink: "#17131F"
  light-ink-muted: "#17131FA8"
  light-accent: "#907AA9"
  light-accent-ink: "#FFFFFF"
  light-positive: "#286983"
  light-danger: "#B4637A"
typography:
  activity-heading:
    fontFamily: "Manrope Variable, Manrope, sans-serif"
    fontSize: "clamp(1.65rem, 3vw, 2.1rem)"
    letterSpacing: "-0.025em"
  page-heading:
    fontFamily: "Manrope Variable, Manrope, sans-serif"
    fontSize: "clamp(1.6rem, 3vw, 2.2rem)"
    letterSpacing: "-0.03em"
  body:
    fontFamily: "Manrope Variable, Manrope, sans-serif"
    lineHeight: 1.5
  section-title:
    fontFamily: "Manrope Variable, Manrope, sans-serif"
    fontSize: "1.05rem"
  field-label:
    fontFamily: "Manrope Variable, Manrope, sans-serif"
    fontSize: "0.9rem"
    fontWeight: 700
rounded:
  card: "24px"
  control: "14px"
spacing:
  1: "4px"
  2: "8px"
  3: "12px"
  4: "16px"
  5: "20px"
  6: "24px"
  7: "32px"
  8: "40px"
---

# Design System: SpeechDock

## Overview

**Creative North Star: "SpeechDock workspace"**

A clean, calm operating surface: real work, state and next actions lead. Lucent 2 supplies the material and shapes, Rosé Pine supplies the two themes, and self-hosted Manrope gives the interface one consistent voice. The waveform identity and restrained Lucide icons support the work without becoming decoration.

**Key Characteristics:**
- Quiet tinted surfaces and separators, with lavender reserved for emphasis and selection.
- Compact, labelled controls and direct feedback instead of decorative statistics.
- Responsive layouts that retain the same tasks on a phone and desktop.

## Colors

### Primary
The dark lavender and light iris accents identify primary actions, selected navigation, progress and focused controls. Accent ink changes with the theme.

### Neutral
Dark uses a deep plum canvas with translucent darker glass and lighter text; light uses a warm paper canvas, pale sheets and dark ink. Muted ink carries supporting descriptions. Sheet and edge values are translucent, so their appearance depends on the canvas beneath them. In light mode, control boundaries are strengthened using a 60% ink color mix rather than relying on the pale sheet edge alone.

### Status
Positive and danger roles distinguish completed or connected states from errors and destructive actions. Status is also written in words; color is not the sole cue.

**The Honest State Rule.** Show actual availability, progress and errors; do not populate the workspace with illustrative activity or invented measurements.

## Typography

Manrope Variable is imported locally with Manrope and sans-serif fallbacks. The same family carries headings, controls and body copy. Page headings use the responsive sizes in the frontmatter; activity and composer headings share one ramp, while Models and Settings use the other. Section titles are compact, field labels are bold, and secondary text uses muted ink. Body line height is 1.5; job preview preserves line breaks while inheriting the interface font.

## Layout

On wide screens the activity workspace has a 14rem navigation rail, flexible history and a 19–25rem composer. A selected job opens its details below the history while the composer remains available. Models and Settings use the rail and main content without the composer. The rail includes shared-engine connectivity on desktop.

At 1150px and below, the workspace rail and composer narrow and outer padding decreases. At 850px and below, content becomes a single column: the composer follows activity, a New job link reaches it, and fixed bottom navigation replaces the rail. Settings groups stack at 650px; upload fields stack at 600px; activity rows and toolbars stack at 520px. At 400px and below, compact 12px panel padding, wrapping and narrower flexible columns protect reflow, including long messages and addresses. Bottom navigation reserves space for the safe-area inset.

## Elevation & Depth

The built application is predominantly tonal and lightly bordered: canvas, sheets and glass-like controls separate regions without prominent drop shadows. Dividers organize rows inside a shared surface. The selected result-preview control uses an inset accent line; focus is expressed by an outline rather than a shadow. The shared token layer includes richer materials, but the current workspace does not depend on live backdrop blur.

## Shapes

Cards use generous rounded corners; controls use the smaller control radius. Inputs, buttons and status markers share this softer geometry, with thin boundaries and restrained fills. Related rows sit within one rounded container instead of becoming a stack of separate floating cards.

## Components

### Buttons and fields

Primary actions use the theme accent and contrasting accent ink. Secondary and danger actions use a glass fill and outlined boundary; danger uses the danger text role. Common form controls are at least 48px high where the Lucent target is applied, while global controls have a 44px minimum. Disabled actions reduce opacity and retain their labels. Inputs and selects are labelled, full-width within their field and use rounded boundaries. Focus-visible outlines are 2px, with an offset; components may use accent rather than the global ink outline.

### Navigation

The brand, labelled Activity, Models and Settings controls, and engine status form the desktop rail. Current navigation has a subtle selected wash and accent icon, and is identified with `aria-current`. The mobile bar retains text labels and icons. Decorative icons are hidden from assistive technology.

### Activity and job details

History is a bordered sheet with a status filter, selectable rows and pagination. Loading, empty, filtered-empty and error states are explicit; errors offer retry. Rows show a filename, task, time and text status; selection is exposed as pressed state. Details show request facts, progress or error when available, result preview and download formats. Cancellation and deletion use an explicit confirmation within the detail surface.

### Upload composer

The composer groups a native file chooser, task, spoken language and model fields in a sheet. Helper copy explains the selected task and model availability; translation offers only registered translation-capable models. Loading, submission, success and failure are visible, and submission is unavailable until required choices are present.

### Models and settings

Models appear as separated entries within a shared card, with descriptive facts, text status, progress and actions appropriate to their state. Shared read-only models are identified as such. Registration, unregistration and file deletion use browser confirmations that describe their effects. Settings use grouped sheets with explanatory introductions, labelled controls, inline feedback and a dedicated save action. Appearance offers System, Light and Dark; the choice is saved in the browser when possible, and System follows the device preference.

### Accessibility and motion

Use semantic sections, labels, named navigation and native controls. Loading and success feedback use status announcements; errors use alerts, and download progress has an accessible label. Preserve keyboard-visible focus, wrapping at narrow widths and a usable layout at increased text size. Reduced-motion preferences suppress animation and transitions globally. These are implementation decisions, not a claim of certification or of a particular performance tier.

## Do's and Don'ts

### Do:
- **Do** keep live states and meaningful next actions clear, including unavailable and empty states.
- **Do** use the shared theme roles and maintain both light and dark treatments.
- **Do** retain labels, visible focus, readable control boundaries and explicit confirmation for consequential actions.

### Don't:
- **Don't** invent job activity or performance figures to fill an empty screen.
- **Don't** make status depend on color alone or hide the consequence of model registration and deletion.
- **Don't** turn the functional workspace into a decorative dashboard.

Source of truth: `ui/src/tokens.css`, `ui/src/app.css` and the Svelte components. The underlying house references are `/data/agent/managed-skills/design-language/languages/lucent/` and `/data/agent/managed-skills/design-language/palettes/rose-pine/`.
