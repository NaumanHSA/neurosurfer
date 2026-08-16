# Architecture diagram — design brief

For `docs/assets/diagrams/neurosurfer-architecture-{light,dark}.jpg`. The README
already points at those two filenames, so replacing them is the whole install.

**Target: 2400 × 1000 px**, readable at 900px wide (GitHub's rendered column).

---

## 1. What the last attempt got wrong

Thin black boxes on cream, every element the same weight, no colour, no icons. It
read as a wireframe rather than a diagram — nothing drew the eye, so the thing
that makes this project different sat in a box the same size as everything else.

The two references get three things right that it did not:

- **Colour carries the sequence.** In the face-capture pipeline each numbered
  step owns a hue and the whole run reads as a progression at a glance.
- **Icons do the naming.** Every card has a line-art glyph, so the shape is
  legible before a word is read.
- **The connectors are alive.** Dotted arrows with endpoint dots, curved
  multi-line streams, a colour gradient along the flow.

Keep the calm, uncluttered *composition* of the last attempt. Add the colour,
the icons, and the movement.

---

## 2. Style

**Ground.** Warm off-white, `#FAF8F4`, with a very soft radial lift toward the
centre. Bottom-left and bottom-right: faint flowing wave contour lines (like the
face-capture reference), 6–8% opacity. Two corners: small dot-matrix grids of
tiny circles, 10% opacity. Never let decoration touch content.

**Cards.** Pure white `#FFFFFF`, radius 16px, a soft diffuse shadow
(`0 4px 16px rgba(0,0,0,0.06)`) — depth from shadow, not from outlines. Where a
card needs an edge, a 1px `#E8E4DC` hairline.

**Type.** Headings in a geometric sans, heavy weight, near-black `#111111`.
Labels and code in a clean monospace. Body copy `#5A5A5A`, small.

**Palette** — one hue per pipeline stage, warm → cool, left to right:

| Stage | Hue | Hex |
|---|---|---|
| 1 Plan | coral | `#FF6B5A` |
| 2 Ground | amber | `#FF9F43` |
| 3 Build | yellow | `#F5C518` |
| 4 Validate | blue | `#3B82F6` |
| 5 Verify | violet | `#8B5CF6` |
| 6 Register | teal | `#14B8A6` |
| Outcome ✓ | green | `#10B981` |
| Outcome ⊘ | slate | `#64748B` |

Icons are **line art only**, 2px stroke, drawn in their stage's hue. No filled
illustrations, no 3D, no photographic elements.

---

## 3. Layout — five regions

### A. Title band (top centre)

Large wordmark **`Neurosurfer`** in heavy sans, flanked by thin coral bracket
accents `[` `]` like the face-capture reference. Above it, small monospace in
coral:

> `agent framework + the architect that builds your workflows`

Under the wordmark, a thin 400px gradient rule running coral → amber → blue →
teal, with two small dots on it.

### B. Left card — "What you write"

Small white card, coral accent bar, gear icon, title **Configuration**. Contains
a monospace snippet:

```
intent: "Read a feedback file, pull out
         the recurring complaints, and
         summarise for the support lead."
```

Below it a small coral pill: `plain English in`

### C. Right card — "What you get"

Small white card, teal accent, lightning icon, title **Runnable workflow**.
Three teal check rows:

- `Grounded — every tool resolved in code`
- `Verified — it ran before it registered`
- `Refuses rather than inventing results`

### D. THE HERO — the Architect pipeline (centre, full width)

**This is the diagram's subject and should occupy the most space.** Six white
cards in a horizontal row, exactly the rhythm of the face-capture pipeline:

Each card has, top to bottom: a **numbered circular badge** in the stage hue
sitting on the card's top edge; a **line-art icon** in that hue inside a soft
tinted rounded square; a bold title; two lines of small grey description.

| # | Icon | Title | Description |
|---|---|---|---|
| 1 | clipboard / list | **Plan** | One structured call<br>steps, inputs, outputs |
| 2 | plug into socket | **Ground** | Capability ladder, in code<br>catalog → MCP registry |
| 3 | blocks assembling | **Build** | A 17-tool agent writes<br>the graph, node by node |
| 4 | checklist shield | **Validate** | A rule table, not a prompt<br>errors block registration |
| 5 | play inside a beaker | **Verify** | Runs it on real fixtures<br>judges the output |
| 6 | package / box | **Register** | A versioned Workflow<br>package on disk |

**Connectors:** between cards, a dashed arrow whose colour *gradients from the
left card's hue into the right card's*, with a small filled dot at the start and
an arrowhead at the end. This is the single most important visual detail — it is
what makes the row read as one flowing pipeline.

**After step 6**, the flow forks into two small outcome cards, side by side:

- ✓ **REGISTERED** — green, `graph.yaml, ready to run`
- ⊘ **REFUSED** — slate, `names the credential or integration that is missing`

### E. The runtime orbit (below the pipeline, left of centre)

Borrowing the Watchtower globe idea, but as a **hexagonal core**, not a planet:

A central dark hexagon badge with the Neurosurfer node-network mark, labelled
**RUNTIME**. Around it, on a dashed circular orbit, six small white circular
icon-nodes connected to the core by thin dashed lines with endpoint dots:

| Position | Icon | Label |
|---|---|---|
| top | flow chart | **Graph Engine**<br>DAG · 11 node kinds |
| top-right | robot | **Agents**<br>AgenticLoop · ReAct · one-shot |
| right | wrench | **Tools**<br>19 built-in + your own |
| bottom-right | plug | **MCP**<br>external servers |
| bottom-left | database + magnifier | **RAG**<br>hybrid · rerank · citations |
| left | cpu | **Providers**<br>one protocol |

A curved dashed stream flows from the **REGISTERED** outcome card into the
runtime core — the workflow arriving at the thing that runs it. A second, fainter
stream enters the core from the top labelled in small monospace
`or write the graph yourself`.

### F. Capability strip (right of the orbit)

Three white cards stacked, each with a coloured icon, matching the face-capture
reference's tech strip:

| Icon | Title | Subtitle |
|---|---|---|
| chips, violet | **Providers** | Anthropic · OpenAI · Gemini · Bedrock |
| database, red | **Vector stores** | Chroma · Qdrant · in-memory |
| waveform, amber | **Observability** | Langfuse · OpenTelemetry |

### G. Footer strip (bottom, full width)

A single row of five small items with tiny grey line icons, separated by thin
vertical dividers — exactly the face-capture reference's bottom row:

`Python 3.11+` │ `Apache-2.0` │ `OpenAI-compatible gateway` │ `Typed & validated`
│ `Runs on a 9B local model`

---

## 4. Dark variant

Same composition. Ground `#0E0E10` with the wave contours at 8% white. Cards
`#18181B` with a `#27272A` hairline and no shadow. Headings `#FAFAFA`, body
`#A1A1AA`. **The eight accent hues stay identical** — they are the only colour in
either version, and keeping them constant is what makes the two read as one pair.

---

## 5. Rules

- **No** gradient-filled backgrounds, glassmorphism, glow, bevels or 3D.
- **No** stock photography, no human figures, no screenshots inside the diagram.
- Line-art icons only, uniform 2px stroke.
- Every arrow terminates in an arrowhead; every connector begins with a dot.
- Whitespace is the layout tool. If it feels crowded, cut region F before
  shrinking the pipeline — the pipeline is the point.

---

## 6. The realistic caveat

**Image models will mangle this much text.** Expect misspelled labels and invented
words. Two ways through:

1. Generate for **composition, colour and icons**, then set the real text over it
   in Figma/Illustrator.
2. Ask for it as **SVG** instead — fully editable, sharp at any size on GitHub and
   PyPI, and the text is guaranteed correct because it *is* text.

If the generated version fights back, say so and the SVG gets hand-built from
this brief.

---

## 7. Prompt

> A wide 2400×1000 technical architecture diagram for a developer tool, warm
> off-white `#FAF8F4` background with faint flowing wave contour lines in the
> bottom corners and small dot-matrix grids in the top corners. Clean editorial
> infographic style: white rounded cards with soft diffuse shadows, thin line-art
> icons with 2px strokes, generous whitespace, flat vector, no 3D, no glow, no
> gradients on backgrounds.
>
> Centre top: bold heading "Neurosurfer" in heavy geometric sans, flanked by thin
> coral bracket marks, with a small coral monospace line above reading "agent
> framework + the architect that builds your workflows", and a thin horizontal
> gradient rule beneath running coral to amber to blue to teal.
>
> The hero element, spanning the full width in the middle: a horizontal pipeline
> of six white rounded cards. Each card has a small numbered circular badge on its
> top edge in its own colour, a line-art icon inside a soft tinted rounded square,
> a bold title, and two short lines of grey description. In order the cards are
> coral "Plan", amber "Ground", yellow "Build", blue "Validate", violet "Verify",
> teal "Register". Between consecutive cards, a dashed arrow whose colour
> gradients from the left card's hue into the right card's, starting with a small
> filled dot and ending in an arrowhead. After the last card the flow forks into
> two small cards: a green one with a checkmark labelled "REGISTERED" and a slate
> one with a crossed circle labelled "REFUSED".
>
> Lower left: a dark hexagonal core badge labelled "RUNTIME" with six small white
> circular icon nodes arranged on a dashed circular orbit around it, each joined
> to the core by a thin dashed line with a dot at each end — labelled Graph
> Engine, Agents, Tools, MCP, RAG, Providers. A curved dashed stream flows from
> the green "REGISTERED" card into this core.
>
> Lower right: three small white cards stacked vertically with coloured line
> icons — a violet chip icon "Providers", a red database icon "Vector stores", an
> amber waveform icon "Observability".
>
> Bottom edge: a single row of five small grey items with tiny line icons
> separated by thin vertical dividers.
>
> Overall: calm, precise, premium developer-documentation aesthetic — colourful
> but restrained, like a well-designed product landing page rather than a
> marketing poster.

For the dark variant, append:

> Dark theme version: near-black `#0E0E10` background, cards `#18181B` with thin
> `#27272A` borders and no shadows, white headings, grey body text. Keep every
> accent colour exactly the same.
