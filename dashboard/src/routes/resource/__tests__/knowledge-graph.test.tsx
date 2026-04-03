import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { KnowledgeGraphPage } from '../knowledge-graph'
import * as memoryApi from '@/lib/memory-api'

const navigateMock = vi.fn()
const toastMock = vi.fn()

vi.mock('@tanstack/react-router', () => ({
  useNavigate: () => navigateMock,
}))

vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({ toast: toastMock }),
}))

vi.mock('@/components/memory/MemoryDeleteDialog', () => ({
  MemoryDeleteDialog: ({
    open,
    preview,
  }: {
    open: boolean
    preview?: { mode?: string; item_count?: number } | null
  }) => (
    open ? <div data-testid="memory-delete-dialog">{`delete:${preview?.mode ?? 'none'}:${preview?.item_count ?? 0}`}</div> : null
  ),
}))

vi.mock('../knowledge-graph/GraphVisualization', () => ({
  GraphVisualization: ({
    graphData,
    onNodeClick,
    onEdgeClick,
  }: {
    graphData: { nodes: Array<{ id: string }>; edges: Array<{ source: string; target: string }> }
    onNodeClick: (event: React.MouseEvent, node: { id: string }) => void
    onEdgeClick: (event: React.MouseEvent, edge: { source: string; target: string }) => void
  }) => (
    <div data-testid="graph-visualization">
      <div>{`nodes:${graphData.nodes.length},edges:${graphData.edges.length}`}</div>
      {graphData.nodes[0] ? (
        <button type="button" onClick={(event) => onNodeClick(event as never, { id: graphData.nodes[0].id })}>
          选择节点
        </button>
      ) : null}
      {graphData.edges[0] ? (
        <button
          type="button"
          onClick={(event) =>
            onEdgeClick(event as never, {
              source: graphData.edges[0].source,
              target: graphData.edges[0].target,
            })}
        >
          选择边
        </button>
      ) : null}
    </div>
  ),
}))

vi.mock('../knowledge-graph/GraphDialogs', () => ({
  NodeDetailDialog: ({
    selectedNodeData,
    nodeDetail,
    onOpenEvidence,
    onDeleteEntity,
  }: {
    selectedNodeData: { id: string } | null
    nodeDetail: { relations?: Array<{ predicate: string }>; paragraphs?: Array<unknown> } | null
    onOpenEvidence?: () => void
    onDeleteEntity?: (options: { includeParagraphs: boolean }) => void
  }) => (
    selectedNodeData ? (
      <div data-testid="node-detail-dialog">
        <div>{`node:${selectedNodeData.id}`}</div>
        <div>{`relations:${nodeDetail?.relations?.[0]?.predicate ?? 'none'}`}</div>
        <div>{`paragraphs:${nodeDetail?.paragraphs?.length ?? 0}`}</div>
        <button type="button" onClick={onOpenEvidence}>切到证据视图</button>
        <button type="button" onClick={() => onDeleteEntity?.({ includeParagraphs: true })}>删除实体</button>
      </div>
    ) : null
  ),
  EdgeDetailDialog: ({
    selectedEdgeData,
    edgeDetail,
    onOpenEvidence,
  }: {
    selectedEdgeData: { source: { id: string }; target: { id: string } } | null
    edgeDetail: { edge?: { predicates?: string[] }; paragraphs?: Array<unknown> } | null
    onOpenEvidence?: () => void
  }) => (
    selectedEdgeData ? (
      <div data-testid="edge-detail-dialog">
        <div>{`edge:${selectedEdgeData.source.id}->${selectedEdgeData.target.id}`}</div>
        <div>{`predicates:${edgeDetail?.edge?.predicates?.join(',') ?? 'none'}`}</div>
        <div>{`paragraphs:${edgeDetail?.paragraphs?.length ?? 0}`}</div>
        <button type="button" onClick={onOpenEvidence}>切到证据视图</button>
      </div>
    ) : null
  ),
  RelationDetailDialog: () => null,
  ParagraphDetailDialog: () => null,
}))

vi.mock('@/lib/memory-api', () => ({
  getMemoryGraph: vi.fn(),
  getMemoryGraphNodeDetail: vi.fn(),
  getMemoryGraphEdgeDetail: vi.fn(),
  previewMemoryDelete: vi.fn(),
  executeMemoryDelete: vi.fn(),
  restoreMemoryDelete: vi.fn(),
}))

describe('KnowledgeGraphPage', () => {
  beforeEach(() => {
    navigateMock.mockReset()
    toastMock.mockReset()
    vi.mocked(memoryApi.getMemoryGraph).mockResolvedValue({
      success: true,
      nodes: [
        { id: 'alpha', name: 'Alpha' },
        { id: 'beta', name: 'Beta' },
      ],
      edges: [
        {
          source: 'alpha',
          target: 'beta',
          weight: 1,
          predicates: ['关联'],
          relation_count: 1,
          evidence_count: 2,
          relation_hashes: ['rel-1'],
          label: '关联',
        },
      ],
      total_nodes: 2,
      total_edges: 1,
    })
    vi.mocked(memoryApi.getMemoryGraphNodeDetail).mockResolvedValue({
      success: true,
      node: { id: 'alpha', type: 'entity', content: 'Alpha', hash: 'entity-1', appearance_count: 3 },
      relations: [
        {
          hash: 'rel-1',
          subject: 'alpha',
          predicate: '关联',
          object: 'beta',
          text: 'alpha 关联 beta',
          confidence: 0.9,
          paragraph_count: 1,
          paragraph_hashes: ['p-1'],
          source_paragraph: 'p-1',
        },
      ],
      paragraphs: [
        {
          hash: 'p-1',
          content: 'Alpha 提到了 Beta',
          preview: 'Alpha 提到了 Beta',
          source: 'demo',
          entity_count: 2,
          relation_count: 1,
          entities: ['Alpha', 'Beta'],
          relations: ['alpha 关联 beta'],
        },
      ],
      evidence_graph: {
        nodes: [
          { id: 'entity:alpha', type: 'entity', content: 'Alpha' },
          { id: 'relation:rel-1', type: 'relation', content: 'alpha 关联 beta' },
          { id: 'paragraph:p-1', type: 'paragraph', content: 'Alpha 提到了 Beta' },
        ],
        edges: [
          { source: 'paragraph:p-1', target: 'entity:alpha', kind: 'mentions', label: '提及', weight: 1 },
          { source: 'paragraph:p-1', target: 'relation:rel-1', kind: 'supports', label: '支撑', weight: 1 },
        ],
        focus_entities: ['alpha'],
      },
    })
    vi.mocked(memoryApi.getMemoryGraphEdgeDetail).mockResolvedValue({
      success: true,
      edge: {
        source: 'alpha',
        target: 'beta',
        weight: 1,
        predicates: ['关联'],
        relation_count: 1,
        evidence_count: 1,
        relation_hashes: ['rel-1'],
        label: '关联',
      },
      relations: [
        {
          hash: 'rel-1',
          subject: 'alpha',
          predicate: '关联',
          object: 'beta',
          text: 'alpha 关联 beta',
          confidence: 0.9,
          paragraph_count: 1,
          paragraph_hashes: ['p-1'],
          source_paragraph: 'p-1',
        },
      ],
      paragraphs: [
        {
          hash: 'p-1',
          content: 'Alpha 提到了 Beta',
          preview: 'Alpha 提到了 Beta',
          source: 'demo',
          entity_count: 2,
          relation_count: 1,
          entities: ['Alpha', 'Beta'],
          relations: ['alpha 关联 beta'],
        },
      ],
      evidence_graph: {
        nodes: [
          { id: 'entity:alpha', type: 'entity', content: 'Alpha' },
          { id: 'entity:beta', type: 'entity', content: 'Beta' },
          { id: 'relation:rel-1', type: 'relation', content: 'alpha 关联 beta' },
        ],
        edges: [
          { source: 'relation:rel-1', target: 'entity:alpha', kind: 'subject', label: '主语', weight: 1 },
          { source: 'relation:rel-1', target: 'entity:beta', kind: 'object', label: '宾语', weight: 1 },
        ],
        focus_entities: ['alpha', 'beta'],
      },
    })
    vi.mocked(memoryApi.previewMemoryDelete).mockResolvedValue({
      success: true,
      mode: 'mixed',
      selector: { entity_hashes: ['entity-1'] },
      counts: { entities: 1, relations: 1, paragraphs: 1 },
      sources: ['demo'],
      items: [{ item_type: 'entity', item_hash: 'entity-1', label: 'Alpha' }],
      item_count: 1,
      dry_run: true,
    } as never)
    vi.mocked(memoryApi.executeMemoryDelete).mockResolvedValue({
      success: true,
      mode: 'mixed',
      operation_id: 'del-1',
      counts: { entities: 1, relations: 1, paragraphs: 1 },
      sources: ['demo'],
      deleted_count: 3,
      deleted_entity_count: 1,
      deleted_relation_count: 1,
      deleted_paragraph_count: 1,
      deleted_source_count: 0,
    } as never)
    vi.mocked(memoryApi.restoreMemoryDelete).mockResolvedValue({ success: true } as never)
  })

  it('renders graph summary and supports empty-result filtering', async () => {
    const user = userEvent.setup()

    render(<KnowledgeGraphPage />)

    expect(await screen.findByText('长期记忆图谱')).toBeInTheDocument()
    expect(screen.getByText(/总节点 2/)).toBeInTheDocument()
    expect(screen.getByTestId('graph-visualization')).toHaveTextContent('nodes:2,edges:1')

    await user.type(screen.getByPlaceholderText('筛选实体名称、节点 ID 或边标签'), 'missing')
    expect(memoryApi.getMemoryGraph).toHaveBeenCalledTimes(1)
    await user.click(screen.getByRole('button', { name: '筛选' }))

    expect(await screen.findByText('还没有可展示的长期记忆图谱')).toBeInTheDocument()
  })

  it('shows empty state when switching to evidence view without a selection', async () => {
    const user = userEvent.setup()

    render(<KnowledgeGraphPage />)

    expect(await screen.findByTestId('graph-visualization')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '证据视图' }))

    expect(await screen.findByText('证据视图还没有可展示的选择')).toBeInTheDocument()
  })

  it('closes node dialog when switching to evidence view and renders evidence graph', async () => {
    const user = userEvent.setup()

    render(<KnowledgeGraphPage />)

    await screen.findByTestId('graph-visualization')
    await user.click(screen.getByRole('button', { name: '选择节点' }))

    expect(await screen.findByTestId('node-detail-dialog')).toHaveTextContent('relations:关联')
    expect(screen.getByTestId('node-detail-dialog')).toHaveTextContent('paragraphs:1')

    await user.click(screen.getByRole('button', { name: '切到证据视图' }))

    await waitFor(() => {
      expect(screen.queryByTestId('node-detail-dialog')).not.toBeInTheDocument()
    })

    await waitFor(() => {
      expect(screen.getByTestId('graph-visualization')).toHaveTextContent('nodes:3,edges:2')
    })
  })

  it('loads edge detail with predicates and support paragraphs', async () => {
    const user = userEvent.setup()

    render(<KnowledgeGraphPage />)

    await screen.findByTestId('graph-visualization')
    await user.click(screen.getByRole('button', { name: '选择边' }))

    expect(await screen.findByTestId('edge-detail-dialog')).toHaveTextContent('predicates:关联')
    expect(screen.getByTestId('edge-detail-dialog')).toHaveTextContent('paragraphs:1')

    await user.click(screen.getByRole('button', { name: '切到证据视图' }))

    await waitFor(() => {
      expect(screen.queryByTestId('edge-detail-dialog')).not.toBeInTheDocument()
    })
  })

  it('opens delete preview dialog from node detail', async () => {
    const user = userEvent.setup()

    render(<KnowledgeGraphPage />)

    await screen.findByTestId('graph-visualization')
    await user.click(screen.getByRole('button', { name: '选择节点' }))
    await screen.findByTestId('node-detail-dialog')

    await user.click(screen.getByRole('button', { name: '删除实体' }))

    await waitFor(() => {
      expect(memoryApi.previewMemoryDelete).toHaveBeenCalled()
    })
    expect(await screen.findByTestId('memory-delete-dialog')).toHaveTextContent('delete:mixed:1')
  })
})
