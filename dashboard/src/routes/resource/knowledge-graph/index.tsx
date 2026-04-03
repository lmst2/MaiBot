import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import type { Edge, Node } from 'reactflow'

import { Database, Network, RefreshCw, Search, SlidersHorizontal } from 'lucide-react'

import { MemoryDeleteDialog } from '@/components/memory/MemoryDeleteDialog'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/hooks/use-toast'
import {
  executeMemoryDelete,
  getMemoryGraph,
  getMemoryGraphEdgeDetail,
  getMemoryGraphNodeDetail,
  previewMemoryDelete,
  restoreMemoryDelete,
  type MemoryDeleteExecutePayload,
  type MemoryDeleteRequestPayload,
  type MemoryEvidenceGraphPayload,
  type MemoryEvidenceParagraphNodeMetadata,
  type MemoryEvidenceRelationNodeMetadata,
  type MemoryGraphEdgeDetailPayload,
  type MemoryGraphNodeDetailPayload,
  type MemoryGraphParagraphDetailPayload,
  type MemoryGraphPayload,
  type MemoryGraphRelationDetailPayload,
} from '@/lib/memory-api'

import {
  EdgeDetailDialog,
  NodeDetailDialog,
  ParagraphDetailDialog,
  RelationDetailDialog,
} from './GraphDialogs'
import { GraphVisualization } from './GraphVisualization'
import type { GraphData, GraphNode, SelectedEdgeData } from './types'

type GraphViewMode = 'entity' | 'evidence'

type DeleteDraft = {
  title: string
  description: string
  request: MemoryDeleteRequestPayload
}

function toEntityGraphData(payload: MemoryGraphPayload): GraphData {
  const nodes: GraphNode[] = (payload.nodes ?? []).map((node) => ({
    id: node.id,
    type: 'entity',
    content: String(node.name ?? node.id),
    metadata: node.attributes ?? {},
  }))
  const edges = (payload.edges ?? []).map((edge) => ({
    source: edge.source,
    target: edge.target,
    weight: Number(edge.weight ?? 1),
    kind: 'relation' as const,
    label: String(edge.label ?? ''),
    relationHashes: edge.relation_hashes ?? [],
    predicates: edge.predicates ?? [],
    relationCount: Number(edge.relation_count ?? edge.relation_hashes?.length ?? 0),
    evidenceCount: Number(edge.evidence_count ?? 0),
  }))
  return { nodes, edges }
}

function toEvidenceGraphData(payload: MemoryEvidenceGraphPayload | null | undefined): GraphData {
  return {
    nodes: (payload?.nodes ?? []).map((node) => ({
      id: node.id,
      type: node.type,
      content: node.content,
      metadata: node.metadata ?? {},
    })),
    edges: (payload?.edges ?? []).map((edge) => ({
      source: edge.source,
      target: edge.target,
      weight: Number(edge.weight ?? 1),
      kind: edge.kind,
      label: edge.label,
    })),
    focusEntities: payload?.focus_entities ?? [],
  }
}

function filterGraphData(graph: GraphData, query: string): GraphData {
  const keyword = query.trim().toLowerCase()
  if (!keyword) {
    return graph
  }

  const matchedNodeIds = new Set(
    graph.nodes
      .filter((node) => node.content.toLowerCase().includes(keyword) || node.id.toLowerCase().includes(keyword))
      .map((node) => node.id),
  )

  const edges = graph.edges.filter((edge) => {
    const label = String(edge.label ?? '').toLowerCase()
    const predicateMatched = (edge.predicates ?? []).some((predicate) => predicate.toLowerCase().includes(keyword))
    const matched =
      matchedNodeIds.has(edge.source) ||
      matchedNodeIds.has(edge.target) ||
      label.includes(keyword) ||
      predicateMatched
    if (matched) {
      matchedNodeIds.add(edge.source)
      matchedNodeIds.add(edge.target)
    }
    return matched
  })

  return {
    nodes: graph.nodes.filter((node) => matchedNodeIds.has(node.id)),
    edges,
    focusEntities: graph.focusEntities,
  }
}

function mergeUniqueRelations(
  nodeDetail: MemoryGraphNodeDetailPayload | null,
  edgeDetail: MemoryGraphEdgeDetailPayload | null,
): MemoryGraphRelationDetailPayload[] {
  const seen = new Set<string>()
  const items: MemoryGraphRelationDetailPayload[] = []
  for (const relation of [...(nodeDetail?.relations ?? []), ...(edgeDetail?.relations ?? [])]) {
    if (seen.has(relation.hash)) {
      continue
    }
    seen.add(relation.hash)
    items.push(relation)
  }
  return items
}

function mergeUniqueParagraphs(
  nodeDetail: MemoryGraphNodeDetailPayload | null,
  edgeDetail: MemoryGraphEdgeDetailPayload | null,
): MemoryGraphParagraphDetailPayload[] {
  const seen = new Set<string>()
  const items: MemoryGraphParagraphDetailPayload[] = []
  for (const paragraph of [...(nodeDetail?.paragraphs ?? []), ...(edgeDetail?.paragraphs ?? [])]) {
    if (seen.has(paragraph.hash)) {
      continue
    }
    seen.add(paragraph.hash)
    items.push(paragraph)
  }
  return items
}

function buildRelationFromMetadata(
  metadata: MemoryEvidenceRelationNodeMetadata | null | undefined,
): MemoryGraphRelationDetailPayload | null {
  const hash = String(metadata?.hash ?? '').trim()
  if (!hash) {
    return null
  }
  const subject = String(metadata?.subject ?? '').trim()
  const predicate = String(metadata?.predicate ?? '').trim()
  const object = String(metadata?.object ?? '').trim()
  const text = String(metadata?.text ?? `${subject} ${predicate} ${object}`).trim()
  return {
    hash,
    subject,
    predicate,
    object,
    text,
    confidence: Number(metadata?.confidence ?? 0),
    paragraph_count: Number(metadata?.paragraph_count ?? 0),
    paragraph_hashes: Array.isArray(metadata?.paragraph_hashes) ? metadata.paragraph_hashes.map(String) : [],
    source_paragraph: '',
  }
}

function buildParagraphFromMetadata(
  metadata: MemoryEvidenceParagraphNodeMetadata | null | undefined,
): MemoryGraphParagraphDetailPayload | null {
  const hash = String(metadata?.hash ?? '').trim()
  if (!hash) {
    return null
  }
  const preview = String(metadata?.preview ?? '').trim()
  return {
    hash,
    content: preview,
    preview,
    source: String(metadata?.source ?? '').trim(),
    updated_at: typeof metadata?.updated_at === 'number' ? metadata.updated_at : null,
    entity_count: Number(metadata?.entity_count ?? 0),
    relation_count: Number(metadata?.relation_count ?? 0),
    entities: [],
    relations: [],
  }
}

export function KnowledgeGraphPage() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [loading, setLoading] = useState(false)
  const [nodeLimit, setNodeLimit] = useState('120')
  const [searchInput, setSearchInput] = useState('')
  const [appliedSearchQuery, setAppliedSearchQuery] = useState('')
  const [viewMode, setViewMode] = useState<GraphViewMode>('entity')
  const [fullGraph, setFullGraph] = useState<GraphData>({ nodes: [], edges: [] })
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const [evidenceGraph, setEvidenceGraph] = useState<GraphData>({ nodes: [], edges: [] })
  const [graphMeta, setGraphMeta] = useState<MemoryGraphPayload | null>(null)
  const [selectedNodeData, setSelectedNodeData] = useState<GraphNode | null>(null)
  const [selectedEdgeData, setSelectedEdgeData] = useState<SelectedEdgeData | null>(null)
  const [nodeDetail, setNodeDetail] = useState<MemoryGraphNodeDetailPayload | null>(null)
  const [edgeDetail, setEdgeDetail] = useState<MemoryGraphEdgeDetailPayload | null>(null)
  const [selectedRelationDetail, setSelectedRelationDetail] = useState<MemoryGraphRelationDetailPayload | null>(null)
  const [selectedRelationMetadata, setSelectedRelationMetadata] = useState<MemoryEvidenceRelationNodeMetadata | null>(null)
  const [selectedParagraphDetail, setSelectedParagraphDetail] = useState<MemoryGraphParagraphDetailPayload | null>(null)
  const [selectedParagraphMetadata, setSelectedParagraphMetadata] = useState<MemoryEvidenceParagraphNodeMetadata | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [deleteDraft, setDeleteDraft] = useState<DeleteDraft | null>(null)
  const [deletePreviewLoading, setDeletePreviewLoading] = useState(false)
  const [deletePreviewError, setDeletePreviewError] = useState<string | null>(null)
  const [deleteResult, setDeleteResult] = useState<MemoryDeleteExecutePayload | null>(null)
  const [deleteExecuting, setDeleteExecuting] = useState(false)
  const [deleteRestoring, setDeleteRestoring] = useState(false)
  const [deletePreview, setDeletePreview] = useState<Awaited<ReturnType<typeof previewMemoryDelete>> | null>(null)

  const allRelationDetails = useMemo(
    () => mergeUniqueRelations(nodeDetail, edgeDetail),
    [edgeDetail, nodeDetail],
  )
  const allParagraphDetails = useMemo(
    () => mergeUniqueParagraphs(nodeDetail, edgeDetail),
    [edgeDetail, nodeDetail],
  )

  const resetDetailSelections = useCallback(() => {
    setSelectedNodeData(null)
    setSelectedEdgeData(null)
    setNodeDetail(null)
    setEdgeDetail(null)
    setSelectedRelationDetail(null)
    setSelectedRelationMetadata(null)
    setSelectedParagraphDetail(null)
    setSelectedParagraphMetadata(null)
  }, [])

  const loadGraph = useCallback(async (options?: { silent?: boolean }) => {
    try {
      setLoading(true)
      const payload = await getMemoryGraph(Number(nodeLimit))
      const nextGraph = toEntityGraphData(payload)
      setGraphMeta(payload)
      setFullGraph(nextGraph)
      setGraphData(filterGraphData(nextGraph, appliedSearchQuery))
      setEvidenceGraph({ nodes: [], edges: [] })
      resetDetailSelections()
      if (!options?.silent) {
        toast({
          title: '图谱已更新',
          description: `当前加载 ${nextGraph.nodes.length} 个节点、${nextGraph.edges.length} 条关系`,
        })
      }
    } catch (error) {
      toast({
        title: '加载失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setLoading(false)
    }
  }, [appliedSearchQuery, nodeLimit, resetDetailSelections, toast])

  useEffect(() => {
    void loadGraph({ silent: true })
  }, [loadGraph])

  const handleSearch = useCallback(() => {
    const nextQuery = searchInput.trim()
    setAppliedSearchQuery(nextQuery)
    const filtered = filterGraphData(fullGraph, nextQuery)
    setGraphData(filtered)
    toast({
      title: nextQuery ? '筛选完成' : '已重置筛选',
      description: `当前显示 ${filtered.nodes.length} 个节点、${filtered.edges.length} 条关系`,
    })
  }, [fullGraph, searchInput, toast])

  const stats = useMemo(
    () => ({
      totalNodes: graphMeta?.total_nodes ?? fullGraph.nodes.length,
      totalEdges: graphMeta?.total_edges ?? fullGraph.edges.length,
      visibleNodes: graphData.nodes.length,
      visibleEdges: graphData.edges.length,
      evidenceNodes: evidenceGraph.nodes.length,
      evidenceEdges: evidenceGraph.edges.length,
    }),
    [
      evidenceGraph.edges.length,
      evidenceGraph.nodes.length,
      fullGraph.edges.length,
      fullGraph.nodes.length,
      graphData.edges.length,
      graphData.nodes.length,
      graphMeta,
    ],
  )

  const openDeleteDialog = useCallback(async (draft: DeleteDraft) => {
    setDeleteDraft(draft)
    setDeletePreview(null)
    setDeleteResult(null)
    setDeletePreviewError(null)
    setDeletePreviewLoading(true)
    try {
      const preview = await previewMemoryDelete(draft.request)
      setDeletePreview(preview)
    } catch (error) {
      setDeletePreviewError(error instanceof Error ? error.message : '删除预览失败')
    } finally {
      setDeletePreviewLoading(false)
    }
  }, [])

  const closeDeleteDialog = useCallback((open: boolean) => {
    if (!open) {
      setDeleteDraft(null)
      setDeletePreview(null)
      setDeleteResult(null)
      setDeletePreviewError(null)
    }
  }, [])

  const executeCurrentDelete = useCallback(async () => {
    if (!deleteDraft) {
      return
    }
    try {
      setDeleteExecuting(true)
      const result = await executeMemoryDelete(deleteDraft.request)
      setDeleteResult(result)
      toast({
        title: result.success ? '删除成功' : '删除失败',
        description: result.success
          ? `操作 ${result.operation_id} 已完成`
          : result.error || '未能执行删除',
        variant: result.success ? 'default' : 'destructive',
      })
      if (result.success) {
        await loadGraph({ silent: true })
      }
    } catch (error) {
      setDeletePreviewError(error instanceof Error ? error.message : '删除失败')
      toast({
        title: '删除失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setDeleteExecuting(false)
    }
  }, [deleteDraft, loadGraph, toast])

  const restoreCurrentDelete = useCallback(async () => {
    if (!deleteResult?.operation_id) {
      return
    }
    try {
      setDeleteRestoring(true)
      await restoreMemoryDelete({
        operation_id: deleteResult.operation_id,
        requested_by: 'knowledge_graph',
      })
      toast({
        title: '恢复成功',
        description: `删除操作 ${deleteResult.operation_id} 已恢复`,
      })
      closeDeleteDialog(false)
      await loadGraph({ silent: true })
    } catch (error) {
      toast({
        title: '恢复失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setDeleteRestoring(false)
    }
  }, [closeDeleteDialog, deleteResult?.operation_id, loadGraph, toast])

  const handleNodeClick = useCallback(async (_: React.MouseEvent, node: Node) => {
    const selected = graphData.nodes.find((item) => item.id === node.id)
    setSelectedNodeData(selected ?? null)
    setSelectedEdgeData(null)
    setEdgeDetail(null)
    setSelectedRelationDetail(null)
    setSelectedParagraphDetail(null)
    if (!selected) {
      return
    }
    try {
      setDetailLoading(true)
      const detail = await getMemoryGraphNodeDetail(selected.id)
      setNodeDetail(detail)
      setEvidenceGraph(toEvidenceGraphData(detail.evidence_graph))
    } catch (error) {
      toast({
        title: '加载节点详情失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setDetailLoading(false)
    }
  }, [graphData.nodes, toast])

  const handleEdgeClick = useCallback(async (_: React.MouseEvent, edge: Edge) => {
    const sourceNode = graphData.nodes.find((nodeItem) => nodeItem.id === edge.source)
    const targetNode = graphData.nodes.find((nodeItem) => nodeItem.id === edge.target)
    const edgeData = graphData.edges.find((item) => item.source === edge.source && item.target === edge.target)
    if (!sourceNode || !targetNode || !edgeData) {
      return
    }
    setSelectedNodeData(null)
    setNodeDetail(null)
    setSelectedRelationDetail(null)
    setSelectedParagraphDetail(null)
    setSelectedEdgeData({
      source: sourceNode,
      target: targetNode,
      edge: edgeData,
    })
    try {
      setDetailLoading(true)
      const detail = await getMemoryGraphEdgeDetail(edge.source, edge.target)
      setEdgeDetail(detail)
      setEvidenceGraph(toEvidenceGraphData(detail.evidence_graph))
    } catch (error) {
      toast({
        title: '加载关系详情失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setDetailLoading(false)
    }
  }, [graphData.edges, graphData.nodes, toast])

  const handleEvidenceNodeClick = useCallback(async (_: React.MouseEvent, node: Node) => {
    const selected = evidenceGraph.nodes.find((item) => item.id === node.id)
    if (!selected) {
      return
    }

    if (selected.type === 'entity') {
      const entityName =
        String((selected.metadata as Record<string, unknown> | undefined)?.entity_name ?? '').trim() || selected.content
      try {
        setDetailLoading(true)
        const detail = await getMemoryGraphNodeDetail(entityName)
        setSelectedNodeData({
          id: detail.node.id,
          type: 'entity',
          content: detail.node.content,
          metadata: { hash: detail.node.hash },
        })
        setSelectedEdgeData(null)
        setNodeDetail(detail)
      } catch (error) {
        toast({
          title: '加载实体详情失败',
          description: error instanceof Error ? error.message : '未知错误',
          variant: 'destructive',
        })
      } finally {
        setDetailLoading(false)
      }
      return
    }

    if (selected.type === 'relation') {
      const metadata = (selected.metadata ?? {}) as MemoryEvidenceRelationNodeMetadata
      const hash = String(metadata.hash ?? '').trim()
      const relation =
        allRelationDetails.find((item) => item.hash === hash) ?? buildRelationFromMetadata(metadata)
      setSelectedRelationMetadata(metadata)
      setSelectedRelationDetail(relation)
      setSelectedParagraphDetail(null)
      return
    }

    if (selected.type === 'paragraph') {
      const metadata = (selected.metadata ?? {}) as MemoryEvidenceParagraphNodeMetadata
      const hash = String(metadata.hash ?? '').trim()
      const paragraph =
        allParagraphDetails.find((item) => item.hash === hash) ?? buildParagraphFromMetadata(metadata)
      setSelectedParagraphMetadata(metadata)
      setSelectedParagraphDetail(paragraph)
      setSelectedRelationDetail(null)
    }
  }, [allParagraphDetails, allRelationDetails, evidenceGraph.nodes, toast])

  const handleOpenNodeEvidence = useCallback(() => {
    setViewMode('evidence')
    setSelectedNodeData(null)
  }, [])

  const handleOpenEdgeEvidence = useCallback(() => {
    setViewMode('evidence')
    setSelectedEdgeData(null)
  }, [])

  const requestDeleteEntity = useCallback(({ includeParagraphs }: { includeParagraphs: boolean }) => {
    const entityHash = String(nodeDetail?.node.hash ?? '').trim()
    if (!entityHash) {
      toast({
        title: '缺少实体标识',
        description: '当前实体没有可用的 hash，无法执行删除。',
        variant: 'destructive',
      })
      return
    }
    void openDeleteDialog({
      title: '删除实体',
      description: '将删除该实体，并自动包含与该实体关联的关系。可按需额外删除支撑段落。',
      request: {
        mode: 'mixed',
        selector: {
          entity_hashes: [entityHash],
          paragraph_hashes: includeParagraphs ? (nodeDetail?.paragraphs ?? []).map((item) => item.hash) : [],
        },
        reason: 'knowledge_graph_delete_entity',
        requested_by: 'knowledge_graph',
      },
    })
  }, [nodeDetail, openDeleteDialog, toast])

  const requestDeleteEdgeGroup = useCallback(({ includeParagraphs }: { includeParagraphs: boolean }) => {
    const relationHashes = edgeDetail?.edge.relation_hashes ?? []
    if (relationHashes.length <= 0) {
      toast({
        title: '缺少关系标识',
        description: '当前关系组没有可用的 relation hash。',
        variant: 'destructive',
      })
      return
    }
    void openDeleteDialog({
      title: '删除关系组',
      description: '将删除这条聚合边对应的全部关系。可按需额外删除支撑段落。',
      request: {
        mode: 'mixed',
        selector: {
          relation_hashes: relationHashes,
          paragraph_hashes: includeParagraphs ? (edgeDetail?.paragraphs ?? []).map((item) => item.hash) : [],
        },
        reason: 'knowledge_graph_delete_edge_group',
        requested_by: 'knowledge_graph',
      },
    })
  }, [edgeDetail, openDeleteDialog, toast])

  const requestDeleteRelation = useCallback(
    (relation: MemoryGraphRelationDetailPayload, includeParagraphs = false) => {
      void openDeleteDialog({
        title: '删除关系',
        description: includeParagraphs ? '将删除这条关系及其支撑段落。' : '将只删除这条关系，保留段落证据。',
        request: {
          mode: 'mixed',
          selector: {
            relation_hashes: [relation.hash],
            paragraph_hashes: includeParagraphs ? relation.paragraph_hashes : [],
          },
          reason: 'knowledge_graph_delete_relation',
          requested_by: 'knowledge_graph',
        },
      })
    },
    [openDeleteDialog],
  )

  const requestDeleteParagraph = useCallback((paragraph: MemoryGraphParagraphDetailPayload) => {
    void openDeleteDialog({
      title: '删除段落证据',
      description: '将删除这段证据，并自动删除失去全部证据的关系。',
      request: {
        mode: 'mixed',
        selector: {
          paragraph_hashes: [paragraph.hash],
        },
        reason: 'knowledge_graph_delete_paragraph',
        requested_by: 'knowledge_graph',
      },
    })
  }, [openDeleteDialog])

  const activeGraph = viewMode === 'entity' ? graphData : evidenceGraph
  const canShowEvidence = Boolean(selectedNodeData || selectedEdgeData || nodeDetail || edgeDetail)

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none border-b bg-card/60 px-6 py-4 backdrop-blur">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-2xl font-bold">长期记忆图谱</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              基于 A_Memorix 的实体关系图与证据视图
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="gap-1">
              <Database className="h-3.5 w-3.5" />
              总节点 {stats.totalNodes}
            </Badge>
            <Badge variant="outline" className="gap-1">
              <Network className="h-3.5 w-3.5" />
              总关系 {stats.totalEdges}
            </Badge>
            <Badge variant="secondary">
              {viewMode === 'entity'
                ? `当前显示 ${stats.visibleNodes} / ${stats.visibleEdges}`
                : `证据视图 ${stats.evidenceNodes} / ${stats.evidenceEdges}`}
            </Badge>
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-3">
          <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
            <div className="flex flex-1 gap-2">
              <Input
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
                onKeyDown={(event) => event.key === 'Enter' && handleSearch()}
                placeholder="筛选实体名称、节点 ID 或边标签"
              />
              <Button onClick={handleSearch} variant="secondary">
                <Search className="mr-2 h-4 w-4" />
                筛选
              </Button>
            </div>

            <div className="flex flex-wrap gap-2">
              <Select value={nodeLimit} onValueChange={setNodeLimit}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="节点上限" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="80">80 节点</SelectItem>
                  <SelectItem value="120">120 节点</SelectItem>
                  <SelectItem value="240">240 节点</SelectItem>
                  <SelectItem value="480">480 节点</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={() => void loadGraph()} disabled={loading}>
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                刷新图谱
              </Button>
              <Button variant="outline" onClick={() => navigate({ to: '/resource/knowledge-base' })}>
                <SlidersHorizontal className="mr-2 h-4 w-4" />
                打开控制台
              </Button>
            </div>
          </div>

          <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as GraphViewMode)}>
            <TabsList className="h-auto flex-wrap justify-start">
              <TabsTrigger value="entity">实体关系图</TabsTrigger>
              <TabsTrigger value="evidence">证据视图</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </div>

      <div className="min-h-0 flex-1 bg-muted/20">
        {viewMode === 'entity' && graphData.nodes.length > 0 ? (
          <GraphVisualization
            graphData={graphData}
            loading={loading}
            onNodeClick={handleNodeClick}
            onEdgeClick={handleEdgeClick}
          />
        ) : viewMode === 'evidence' && activeGraph.nodes.length > 0 ? (
          <GraphVisualization
            graphData={activeGraph}
            loading={detailLoading}
            onNodeClick={handleEvidenceNodeClick}
            onEdgeClick={() => undefined}
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="max-w-xl rounded-xl border bg-background p-8 text-center shadow-sm">
              {viewMode === 'entity' ? (
                <>
                  <h2 className="text-lg font-semibold">还没有可展示的长期记忆图谱</h2>
                  <p className="mt-2 text-sm text-muted-foreground">
                    先在长期记忆控制台里完成导入或记忆生成，再回来查看关系网络。
                  </p>
                  <Button className="mt-4" onClick={() => navigate({ to: '/resource/knowledge-base' })}>
                    前往长期记忆控制台
                  </Button>
                </>
              ) : (
                <>
                  <h2 className="text-lg font-semibold">证据视图还没有可展示的选择</h2>
                  <p className="mt-2 text-sm text-muted-foreground">
                    先在实体关系图里点击某个实体或边，再切换到证据视图查看 paragraph → relation/entity 的牵引。
                  </p>
                  <div className="mt-4 flex justify-center gap-2">
                    <Button variant="outline" onClick={() => setViewMode('entity')}>
                      返回实体关系图
                    </Button>
                    {canShowEvidence && (
                      <Button onClick={() => setViewMode('evidence')}>刷新证据视图</Button>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      <NodeDetailDialog
        open={Boolean(selectedNodeData)}
        onOpenChange={(open) => !open && setSelectedNodeData(null)}
        selectedNodeData={selectedNodeData}
        nodeDetail={nodeDetail}
        loading={detailLoading}
        onOpenEvidence={handleOpenNodeEvidence}
        onDeleteEntity={requestDeleteEntity}
        onDeleteRelation={(relation) => requestDeleteRelation(relation)}
        onDeleteParagraph={requestDeleteParagraph}
      />
      <EdgeDetailDialog
        open={Boolean(selectedEdgeData)}
        onOpenChange={(open) => !open && setSelectedEdgeData(null)}
        selectedEdgeData={selectedEdgeData}
        edgeDetail={edgeDetail}
        loading={detailLoading}
        onOpenEvidence={handleOpenEdgeEvidence}
        onDeleteEdgeGroup={requestDeleteEdgeGroup}
        onDeleteRelation={(relation) => requestDeleteRelation(relation)}
        onDeleteParagraph={requestDeleteParagraph}
      />
      <RelationDetailDialog
        open={Boolean(selectedRelationDetail)}
        onOpenChange={(open) => !open && setSelectedRelationDetail(null)}
        relation={selectedRelationDetail}
        metadata={selectedRelationMetadata}
        onDeleteRelation={(relation, includeParagraphs) => requestDeleteRelation(relation, includeParagraphs)}
      />
      <ParagraphDetailDialog
        open={Boolean(selectedParagraphDetail)}
        onOpenChange={(open) => !open && setSelectedParagraphDetail(null)}
        paragraph={selectedParagraphDetail}
        metadata={selectedParagraphMetadata}
        onDeleteParagraph={requestDeleteParagraph}
      />
      <MemoryDeleteDialog
        open={Boolean(deleteDraft)}
        onOpenChange={closeDeleteDialog}
        title={deleteDraft?.title ?? '删除预览'}
        description={deleteDraft?.description}
        preview={deletePreview}
        result={deleteResult}
        loadingPreview={deletePreviewLoading}
        executing={deleteExecuting}
        restoring={deleteRestoring}
        error={deletePreviewError}
        onExecute={executeCurrentDelete}
        onRestore={restoreCurrentDelete}
      />
    </div>
  )
}
