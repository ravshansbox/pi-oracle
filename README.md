# @ravshansbox/pi-oracle

Second-opinion model extension for pi.

## Usage

After an assistant answers, run:

```text
/oracle
```

Oracle opens a searchable picker containing the authenticated models available in pi, excluding the model that produced the latest answer. The selected model independently reviews the conversation and latest answer, then adds its opinion to the current context.

The selected model is remembered per reviewed model. The next time `/oracle` reviews that model, the previous Oracle model is preselected.

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
