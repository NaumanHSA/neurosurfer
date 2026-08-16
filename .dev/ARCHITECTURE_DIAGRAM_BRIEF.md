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

**No product wordmark.** This sits under a `## Architecture` heading in the
README, so a big "Neurosurfer" repeats what the reader just read and makes the
diagram look like a second banner. Delete it, and the bracket accents with it.

What remains, centred: one line of monospace in coral,

> `agent framework + the architect that builds your workflows`

and beneath it the thin 400px gradient rule running coral → amber → blue → teal
with two small dots on it.

The band is now short, so **close the gap** — raise the two side cards to sit
level with the tagline and bring the whole composition up. Do not leave the
freed space empty; the diagram should get taller cards and more breathing room
around the pipeline instead.

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

The description is **two separate lines**, stacked. Do not join them with a `/`
or any other separator — "One structured call / steps, inputs, outputs" reads as
a path or a fraction. Line one, line break, line two.

| # | Icon | Title | Description |
|---|---|---|---|
| 1 | clipboard / list | **Plan** | One structured call<br>steps, inputs, outputs |
| 2 | plug into socket | **Ground** | Capability ladder, in code<br>catalog → MCP registry |
| 3 | blocks assembling | **Build** | An agent with 15+ tools<br>writes the graph, node by node |
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
| right | wrench | **Tools**<br>15+ built-in · plus your own |
| bottom-right | plug | **MCP**<br>external servers |
| bottom-left | database + magnifier | **RAG**<br>hybrid · rerank · citations |
| left | cpu | **Providers**<br>one protocol |

**Two streams enter the core, and both must land on it.**

1. A curved dashed **green** stream from the **REGISTERED** card, ending in an
   arrowhead that *touches the hexagon's edge* — the workflow arriving at the
   thing that runs it. It must not stop in open space short of the core.
2. A separate **grey** stream labelled `or write the graph yourself`, entering
   the core from above. **It must not originate anywhere on the Architect
   pipeline.** It is the second door — a person writing the graph by hand,
   bypassing the Architect entirely — so it starts at the top edge of the canvas
   and comes straight down. A version that branched off the *Validate* card said
   the opposite of what is meant.

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
│ `Runs on local models`

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
- **State counts as a floor — `15+`, not `17`.** An image is the most expensive
  thing in the repo to correct, so nothing in it should depend on a number that
  moves when someone adds a tool. The exception is `11 node kinds`: that is a
  designed API surface rather than a growing collection, and `11+` would
  understate it.

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
> Centre top, with NO product name and no wordmark: a single small coral
> monospace line reading "agent framework + the architect that builds your
> workflows", and beneath it a thin horizontal gradient rule running coral to
> amber to blue to teal with two small dots on it. Two small white cards flank
> it at the same height, one on each side.
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
> Engine, Agents, Tools, MCP, RAG, Providers. A curved dashed green stream flows
> from the green "REGISTERED" card and ends in an arrowhead touching the edge of
> the hexagon. A separate grey dashed arrow comes straight down from the top of
> the canvas into the same hexagon, labelled "or write the graph yourself" — it
> does not touch the six-card pipeline anywhere.
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


---

## 8. Second pass — what to change from the first generated version

The first attempt was close, and most of it should be kept: the pipeline rhythm,
the colour progression, the icon set, the orbit, the footer. Six changes.

| # | Change | Why |
|---|---|---|
| 1 | **Delete the "Neurosurfer" wordmark and its bracket accents.** Keep the coral tagline and the gradient rule. | It sits under a `## Architecture` heading, so the name is already on screen. It made a diagram look like a banner. |
| 2 | **Footer: "Runs on local models"**, not "Runs on a 9B local model". | The size is a detail of one test run, not a property of the framework. |
| 3 | **Build card: "An agent with 15+ tools".** Tools orbit node: **"15+ built-in · plus your own"**. | "17-tool" was wrong — the belt is 17 base plus `test_workflow` plus `web_search`. Counts stated as a floor survive the next tool anyone adds; an exact number is a promise the diagram has to keep and a regenerated image to keep it. |
| 4 | **The grey `or write the graph yourself` arrow must not start at the Validate card.** Bring it straight down from the top of the canvas into the runtime hexagon. | It is the *second door* — a human writing the graph, bypassing the Architect. Branching it off a pipeline stage says the opposite. |
| 5 | **The green stream from REGISTERED must terminate on the hexagon**, arrowhead touching its edge. | In the first version it stopped in open space beside the Providers card, so the workflow never visibly arrives anywhere. |
| 6 | **Card descriptions on two stacked lines, no `/` separator.** | "One structured call / steps, inputs, outputs" reads as a path or a fraction. |

### Checked and correct — leave alone

`11 node kinds` · `Chroma · Qdrant · in-memory` ·
`Anthropic · OpenAI · Gemini · Bedrock` · `Langfuse · OpenTelemetry` · the
REGISTERED / REFUSED fork and its wording. All verified against the code.

### Spellings a generator gets wrong

`Anthropic` (not Anthropric) · `Qdrant` (not Qdrant/Quadrant) · `OpenTelemetry`
(one word, capital T) · `Langfuse` (lowercase f) · `AgenticLoop` (one word, two
capitals) · `graph.yaml` (lowercase).
