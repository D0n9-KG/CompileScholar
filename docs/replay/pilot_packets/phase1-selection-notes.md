# Phase 1 Packet Selection Notes

## Packet Identity

- `packet_id`: `jamming_frictionless_packings_2010_packet_01`
- `topic_scope_candidate`: `jamming transition in frictionless sphere packings near point J`
- `cutoff_year`: `2010`
- `packet_status`: `reviewed`

## Why This Topic

This packet is intentionally built around the classic `point J` / frictionless-packing route because it satisfies the main Phase 1 needs:

- the historical boundary is easy to state and audit
- there is a clear main route plus at least one nearby alternative route
- the packet can carry a real critique signal instead of only supportive papers
- the topic is narrow enough that replay failures will be interpretable

This is a bounded pilot packet, not a domain-complete packet.

## Why The Packet Is Bounded

The packet does **not** try to cover all of jamming physics. It is limited to:

- frictionless sphere packings and point-J style formulations
- nearby conceptual predecessors and route-adjacent alternatives visible by 2010
- papers that help compile one route state, one why-now judgment, and one route comparison

The packet explicitly excludes:

- post-2010 shear-jamming and later finite-size syntheses as packet inputs
- later review articles as training inputs
- metaphorical or off-domain uses of the word `jamming`

## Intended Role Composition

- `core_method`: 3 papers
- `resource_or_benchmark`: 1 paper
- `limitation_or_critique`: 1 paper
- `survey_or_review`: 1 paper
- `alternative_route`: 1 paper

This is smaller than the long-term recommended packet size. That is deliberate for Phase 1:

- the goal is to test the replay workflow on a real packet
- not to maximize paper count on the first pass

## Included Papers And Rationale

| Paper ID | Year | Role | Why Included |
|----------|------|------|--------------|
| `777` | 1998 | `survey_or_review` | Early conceptual framing of jamming / fragile matter used to set packet boundary |
| `1742` | 2000 | `resource_or_benchmark` | Provides the MRJ / random-close-packing reference frame and protocol benchmark |
| `1746` | 2002 | `core_method` | Establishes the numerical frictionless-packing route |
| `1591` | 2003 | `core_method` | Centers the route on zero-temperature / zero-stress point J |
| `820` | 2006 | `core_method` | Adds linear-response scaling signals near jamming |
| `814` | 2010 | `alternative_route` | Keeps frictional packing route visible as a comparison target |
| `1647` | 2010 | `limitation_or_critique` | Preserves protocol-dependence and non-unique-density critique |

## Excluded Items And Leakage Discipline

The packet deliberately records visible exclusions:

- `1628` (2011): relevant shear-flow probe, but after cutoff
- `1654` (2014): later finite-size synthesis, audit-only
- `1632` (2016): later hard-sphere equilibrium sampling result, audit-only
- `770` (2019): later review, audit-only
- `1845` (2024): off-topic deep-learning analogy

Leakage discipline for this packet:

- no post-2010 paper may enter packet input
- later reviews are audit-only and must not be fed to the route compiler as if they were contemporaneous evidence
- hindsight judgments about which route “won” must stay out of the packet and only appear later in replay review artifacts

## Known Gaps

- trace files are local-only and are **not** committed in the repo
- `trace_id` fields are committed as canonical target ids, but local trace export availability still needs to be verified when running the pilot
- `L1` snapshot refs are logical placeholders for the replay contract and still need real historical asset backfill
- the packet is intentionally small, so we should expect `quality_flags` rather than a clean green replay

## What This Packet Should Support

This packet is intended to support:

- one first real `RouteState` replay
- one first `WhyNowCase`
- one explicit comparison between the frictionless point-J route and a frictional alternative
- one first failure inventory showing what the current `L2 -> L3/L4` stack still misses

## What It Should Not Be Used To Claim

This packet should **not** be treated as:

- a complete history of jamming research
- a final, production-grade route packet
- proof that the current `L2` layer is already sufficient

Its purpose is to expose the next real bottlenecks with an auditable packet boundary.
