import { describe, expect, test } from 'vitest'

import { buildPaperFlowPositions, type SignalGraphEdge, type SignalGraphNode } from '../src/components/SignalGraph'

function radiusForKind(kind: SignalGraphNode['kind']): number {
  if (kind === 'root') return 46
  if (kind === 'cluster') return 134
  if (kind === 'move') return 20
  if (kind === 'anchor') return 17
  return 28
}

describe('buildPaperFlowPositions', () => {
  test('keeps dense move clusters separated without node overlap', () => {
    const nodes: SignalGraphNode[] = [
      { id: 'paper:1', label: 'Center Paper', kind: 'root', weight: 1 },
    ]
    const edges: SignalGraphEdge[] = []
    const anchorCounts = [8, 7, 6, 5, 4, 4]

    anchorCounts.forEach((anchorCount, moveIndex) => {
      const clusterId = `cluster:${moveIndex + 1}`
      const moveId = `move:${moveIndex + 1}`
      nodes.push({
        id: clusterId,
        label: `Cluster ${moveIndex + 1}`,
        kind: 'cluster',
        weight: 0.8,
      })
      nodes.push({
        id: moveId,
        label: `Move ${moveIndex + 1}`,
        kind: 'move',
        weight: 0.7,
      })
      edges.push({
        id: `paper:1->${moveId}`,
        source: 'paper:1',
        target: moveId,
        kind: 'supports',
        weight: 0.7,
      })

      for (let anchorIndex = 0; anchorIndex < anchorCount; anchorIndex += 1) {
        const anchorId = `anchor:${moveIndex + 1}-${anchorIndex + 1}`
        nodes.push({
          id: anchorId,
          label: `Anchor ${moveIndex + 1}-${anchorIndex + 1}`,
          kind: 'anchor',
          weight: 0.55,
        })
        edges.push({
          id: `${moveId}->${anchorId}`,
          source: moveId,
          target: anchorId,
          kind: 'evidenced_by',
          weight: 0.55,
        })
      }
    })

    const positions = buildPaperFlowPositions(nodes, edges, 1360, 920)

    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i]
        const b = nodes[j]
        if (a.kind === 'cluster' || b.kind === 'cluster') continue
        const pa = positions.get(a.id)
        const pb = positions.get(b.id)
        expect(pa, `${a.id} should have a position`).toBeTruthy()
        expect(pb, `${b.id} should have a position`).toBeTruthy()
        const distance = Math.hypot((pa?.x ?? 0) - (pb?.x ?? 0), (pa?.y ?? 0) - (pb?.y ?? 0))
        const minDistance = radiusForKind(a.kind) + radiusForKind(b.kind)
        expect(distance, `${a.id} and ${b.id} should not overlap`).toBeGreaterThan(minDistance)
      }
    }

    const clusterCenters = anchorCounts.map((_, moveIndex) => {
      const clusterId = `cluster:${moveIndex + 1}`
      const moveId = `move:${moveIndex + 1}`
      const clusterPoint = positions.get(clusterId)
      const movePoint = positions.get(moveId)
      expect(clusterPoint, `${clusterId} should have a position`).toBeTruthy()
      expect(movePoint, `${moveId} should have a position`).toBeTruthy()
      expect(
        Math.hypot((clusterPoint?.x ?? 0) - (movePoint?.x ?? 0), (clusterPoint?.y ?? 0) - (movePoint?.y ?? 0)),
        `${clusterId} should anchor ${moveId}`,
      ).toBeLessThan(28)
      const members = nodes.filter(
        (node) => node.id === moveId || (node.kind === 'anchor' && node.id.startsWith(`anchor:${moveIndex + 1}-`)),
      )
      const coords = members.map((node) => positions.get(node.id)).filter(Boolean) as Array<{ x: number; y: number }>
      const center = coords.reduce(
        (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
        { x: 0, y: 0 },
      )
      return {
        id: clusterId,
        x: clusterPoint?.x ?? center.x / coords.length,
        y: clusterPoint?.y ?? center.y / coords.length,
      }
    })

    const rootPoint = positions.get('paper:1')
    expect(rootPoint, 'paper:1 should have a position').toBeTruthy()
    for (const center of clusterCenters) {
      const distance = Math.hypot((rootPoint?.x ?? 0) - center.x, (rootPoint?.y ?? 0) - center.y)
      expect(distance, `paper:1 and ${center.id} should stay separated`).toBeGreaterThan(
        radiusForKind('root') + radiusForKind('cluster') + 28,
      )
    }

    for (let i = 0; i < clusterCenters.length; i += 1) {
      for (let j = i + 1; j < clusterCenters.length; j += 1) {
        const a = clusterCenters[i]
        const b = clusterCenters[j]
        const distance = Math.hypot(a.x - b.x, a.y - b.y)
        expect(distance, `${a.id} and ${b.id} clusters should stay separated`).toBeGreaterThan(
          radiusForKind('cluster') * 2 + 24,
        )
      }
    }
  })
})
