# @ravshansbox/pi-oracle

Second-opinion model extension for pi.

## Usage

After an assistant answers, run:

```text
/oracle
```

Add an optional review instruction after the command:

```text
/oracle what do you think? answer short
```

Oracle opens a searchable picker containing the authenticated models available in pi, excluding the model that produced the latest answer. Type to filter by model ID, provider, or display name. Press `Tab` to cycle through the selected model's supported thinking levels, then press `Enter` to run the review.

The current Pi thinking level is used by default and clamped to the selected model's capabilities. The selected model is remembered per reviewed model. The next time `/oracle` reviews that model, the previous Oracle model is preselected.

Oracle honours explicit output constraints from the latest user message and optional command instruction. Requests for a short or brief answer suppress the default detailed sections.

Oracle sends recent conversation text, tool calls, and tool results to the selected model's provider. Hidden thinking is excluded. Images are not forwarded, so image-dependent answers may receive an incomplete review. Nested-call usage appears in the expanded Oracle card but is not included in pi's session totals.

## Model pairings

Pairings are stored as a flat map in `<Pi agent directory>/oracle.json` (`~/.pi/agent/oracle.json` by default):

```json
{
  "anthropic/claude-opus-5": "openai-codex/gpt-5.6-sol",
  "openai-codex/gpt-5.6-sol": "anthropic/claude-opus-5"
}
```

Each direction is independent and updates when a different Oracle model is selected.

## Installation

```bash
pi install npm:@ravshansbox/pi-oracle
```

### Project-local installation

```bash
pi install -l npm:@ravshansbox/pi-oracle
```

### Install from Git

```bash
pi install git:git@github.com:ravshansbox/pi-oracle.git
```

## Development

```bash
npm install
npm run typecheck
```
