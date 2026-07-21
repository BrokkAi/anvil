---
title: Sessions and Context
description: Understand durable history, client-owned controls, compaction, and lifecycle operations.
---

Anvil persists ACP sessions so a client can list, load, resume, fork, close, or delete work according to the protocol surface it implements.

## Durable State

Persisted sessions retain raw conversation turns, tool exchanges, task metadata, and Anvil compaction checkpoints. The exact client-visible presentation and lifecycle controls belong to the ACP client.

Model, behavior, reasoning effort, service tier, and permission selections are client-owned live controls. They must not be treated as install-wide Anvil preferences; clients resubmit desired values for new or restored sessions. `/setup timeout` is a separate in-memory session override and is not an ACP configuration selector.

## Context Reporting

```text
/context
```

The report describes the current model-facing prompt shape and token estimate. It helps explain why a compaction threshold is near, but it is not a provider billing report.

## Compaction

```text
/compact
```

Compaction replaces an older dynamic model-history prefix with a cumulative state snapshot, retains a recent exact tail, and pins the current plan. Oversized compactor input is chunked and reduced before the final snapshot pass.

Raw turns remain in the session archive for replay and rewind. Compaction changes what is sent to the model; it does not erase the underlying verbatim history. Automatic compaction can run before a new user turn or between completed tool exchanges when the context budget requires it.

## Retention and Trust

Session archives are local data. They may contain prompts, model responses, tool results, pasted credentials from text-only setup flows, and source excerpts. Protect, retain, back up, and delete them according to the sensitivity of the project. See [Data and Trust Boundaries](/data-boundaries/).
