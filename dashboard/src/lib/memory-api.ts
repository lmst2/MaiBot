import type { PluginConfigSchema } from '@/lib/plugin-api'

import { getApiBaseUrl } from './api-base'
import { isElectron } from './runtime'

async function getMemoryApiBase(): Promise<string> {
  if (isElectron()) {
    const base = await getApiBaseUrl()
    return base ? `${base}/api/webui/memory` : '/api/webui/memory'
  }
  return import.meta.env.VITE_API_BASE_URL
    ? `${import.meta.env.VITE_API_BASE_URL}/memory`
    : '/api/webui/memory'
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${await getMemoryApiBase()}${path}`, init)
  if (!response.ok) {
    let detail = `${response.status}`
    try {
      const payload = await response.json()
      detail = String(payload?.detail ?? payload?.error ?? detail)
    } catch {
      // ignore json parsing fallback
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export interface MemoryGraphNodePayload {
  id: string
  name: string
  attributes?: Record<string, unknown>
}

export interface MemoryGraphEdgePayload {
  source: string
  target: string
  weight: number
  relation_hashes?: string[]
  predicates?: string[]
  relation_count?: number
  evidence_count?: number
  label?: string
}

export interface MemoryGraphPayload {
  success: boolean
  nodes: MemoryGraphNodePayload[]
  edges: MemoryGraphEdgePayload[]
  total_nodes: number
  total_edges: number
}

export interface MemoryGraphRelationDetailPayload {
  hash: string
  subject: string
  predicate: string
  object: string
  text: string
  confidence: number
  paragraph_count: number
  paragraph_hashes: string[]
  source_paragraph: string
}

export interface MemoryGraphParagraphDetailPayload {
  hash: string
  content: string
  preview: string
  source: string
  created_at?: number | null
  updated_at?: number | null
  entity_count: number
  relation_count: number
  entities: string[]
  relations: string[]
}

export interface MemoryEvidenceGraphNodePayload {
  id: string
  type: 'entity' | 'relation' | 'paragraph'
  content: string
  metadata?: MemoryEvidenceGraphNodeMetadata
}

export interface MemoryEvidenceGraphEdgePayload {
  source: string
  target: string
  kind: 'mentions' | 'supports' | 'subject' | 'object'
  label: string
  weight: number
}

export interface MemoryEvidenceGraphPayload {
  nodes: MemoryEvidenceGraphNodePayload[]
  edges: MemoryEvidenceGraphEdgePayload[]
  focus_entities: string[]
}

export interface MemoryEvidenceEntityNodeMetadata extends Record<string, unknown> {
  entity_name?: string
}

export interface MemoryEvidenceRelationNodeMetadata extends Record<string, unknown> {
  hash?: string
  subject?: string
  predicate?: string
  object?: string
  confidence?: number
  paragraph_count?: number
  paragraph_hashes?: string[]
  text?: string
}

export interface MemoryEvidenceParagraphNodeMetadata extends Record<string, unknown> {
  hash?: string
  source?: string
  updated_at?: number | null
  entity_count?: number
  relation_count?: number
  preview?: string
}

export type MemoryEvidenceGraphNodeMetadata =
  | MemoryEvidenceEntityNodeMetadata
  | MemoryEvidenceRelationNodeMetadata
  | MemoryEvidenceParagraphNodeMetadata
  | Record<string, unknown>

export interface MemoryGraphNodeDetailPayload {
  success: boolean
  node: {
    id: string
    type: 'entity'
    content: string
    hash?: string
    appearance_count?: number
  }
  relations: MemoryGraphRelationDetailPayload[]
  paragraphs: MemoryGraphParagraphDetailPayload[]
  evidence_graph: MemoryEvidenceGraphPayload
}

export interface MemoryGraphEdgeDetailPayload {
  success: boolean
  edge: MemoryGraphEdgePayload
  relations: MemoryGraphRelationDetailPayload[]
  paragraphs: MemoryGraphParagraphDetailPayload[]
  evidence_graph: MemoryEvidenceGraphPayload
}

export interface MemoryRuntimeConfigPayload {
  success: boolean
  config: Record<string, unknown>
  data_dir: string
  embedding_dimension: number
  auto_save: boolean
  relation_vectors_enabled: boolean
  runtime_ready: boolean
  embedding_degraded: boolean
  embedding_degraded_reason: string
  embedding_degraded_since?: number | null
  embedding_last_check?: number | null
  paragraph_vector_backfill_pending: number
  paragraph_vector_backfill_running: number
  paragraph_vector_backfill_failed: number
  paragraph_vector_backfill_done: number
}

export interface MemoryRuntimeSelfCheckPayload {
  success: boolean
  report?: Record<string, unknown>
  error?: string
}

export interface MemoryConfigPayload {
  success: boolean
  config: Record<string, unknown>
  path: string
}

export interface MemoryRawConfigPayload {
  success: boolean
  config: string
  path: string
}

export interface MemoryConfigSchemaPayload {
  success: boolean
  schema: PluginConfigSchema
  path: string
}

export interface MemoryImportGuidePayload {
  success: boolean
  content: string
  source?: string
  path?: string
  settings?: Record<string, unknown>
}

export interface MemoryTaskPayload {
  task_id?: string
  status?: string
  mode?: string
  created_at?: number
  updated_at?: number
  [key: string]: unknown
}

export interface MemoryTaskListPayload {
  success: boolean
  items: MemoryTaskPayload[]
  count?: number
  settings?: Record<string, unknown>
}

export interface MemoryTuningProfilePayload {
  success: boolean
  profile?: Record<string, unknown>
  settings?: Record<string, unknown>
  toml?: string
}

export interface MemoryDeleteCountsPayload {
  relations?: number
  paragraphs?: number
  entities?: number
  sources?: number
  requested_sources?: number
  matched_sources?: number
  [key: string]: number | undefined
}

export interface MemoryDeletePreviewItemPayload {
  item_type: string
  item_hash: string
  item_key?: string
  label?: string
  preview?: string
  source?: string
}

export interface MemoryDeleteRequestPayload {
  mode: string
  selector: Record<string, unknown> | string
  reason?: string
  requested_by?: string
}

export interface MemoryDeletePreviewPayload {
  success: boolean
  mode: string
  selector: Record<string, unknown> | string
  counts: MemoryDeleteCountsPayload
  sources: string[]
  items: MemoryDeletePreviewItemPayload[]
  item_count: number
  dry_run?: boolean
  requested_source_count?: number
  matched_source_count?: number
  vector_ids?: string[]
  error?: string
}

export interface MemoryDeleteExecutePayload {
  success: boolean
  mode: string
  operation_id: string
  counts: MemoryDeleteCountsPayload
  sources: string[]
  deleted_count: number
  deleted_entity_count: number
  deleted_relation_count: number
  deleted_paragraph_count: number
  deleted_source_count: number
  deleted_vector_count?: number
  requested_source_count?: number
  matched_source_count?: number
  error?: string
  deleted?: boolean | number
}

export interface MemoryDeleteOperationItemPayload {
  item_type: string
  item_hash: string
  item_key?: string
  payload?: Record<string, unknown>
  created_at?: number
}

export interface MemoryDeleteOperationPayload {
  operation_id: string
  mode: string
  selector?: Record<string, unknown> | string
  reason?: string | null
  requested_by?: string | null
  status?: string
  created_at?: number
  restored_at?: number | null
  summary?: Record<string, unknown>
  items?: MemoryDeleteOperationItemPayload[]
}

export interface MemoryDeleteOperationListPayload {
  success: boolean
  items: MemoryDeleteOperationPayload[]
  count?: number
  error?: string
}

export interface MemoryDeleteOperationDetailPayload {
  success: boolean
  operation?: MemoryDeleteOperationPayload | null
  error?: string
}

export interface MemorySourceItemPayload {
  source: string
  paragraph_count?: number
  relation_count?: number
  episode_rebuild_blocked?: boolean
  [key: string]: unknown
}

export interface MemorySourceListPayload {
  success: boolean
  items: MemorySourceItemPayload[]
  count: number
}

export async function getMemoryGraph(limit: number = 120): Promise<MemoryGraphPayload> {
  return requestJson<MemoryGraphPayload>(`/graph?limit=${limit}`)
}

export async function getMemoryGraphNodeDetail(
  nodeId: string,
  options?: {
    relationLimit?: number
    paragraphLimit?: number
    evidenceNodeLimit?: number
  },
): Promise<MemoryGraphNodeDetailPayload> {
  const params = new URLSearchParams({
    node_id: nodeId,
    relation_limit: String(options?.relationLimit ?? 20),
    paragraph_limit: String(options?.paragraphLimit ?? 20),
    evidence_node_limit: String(options?.evidenceNodeLimit ?? 80),
  })
  return requestJson<MemoryGraphNodeDetailPayload>(`/graph/node-detail?${params.toString()}`)
}

export async function getMemoryGraphEdgeDetail(
  source: string,
  target: string,
  options?: {
    paragraphLimit?: number
    evidenceNodeLimit?: number
  },
): Promise<MemoryGraphEdgeDetailPayload> {
  const params = new URLSearchParams({
    source,
    target,
    paragraph_limit: String(options?.paragraphLimit ?? 20),
    evidence_node_limit: String(options?.evidenceNodeLimit ?? 80),
  })
  return requestJson<MemoryGraphEdgeDetailPayload>(`/graph/edge-detail?${params.toString()}`)
}

export async function previewMemoryDelete(
  payload: MemoryDeleteRequestPayload,
): Promise<MemoryDeletePreviewPayload> {
  return requestJson<MemoryDeletePreviewPayload>('/delete/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function executeMemoryDelete(
  payload: MemoryDeleteRequestPayload,
): Promise<MemoryDeleteExecutePayload> {
  return requestJson<MemoryDeleteExecutePayload>('/delete/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function restoreMemoryDelete(payload: {
  operation_id: string
  mode?: string
  selector?: Record<string, unknown> | string
  reason?: string
  requested_by?: string
}): Promise<Record<string, unknown>> {
  return requestJson('/delete/restore', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function getMemoryDeleteOperations(
  limit: number = 20,
  mode: string = '',
): Promise<MemoryDeleteOperationListPayload> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (mode.trim()) {
    params.set('mode', mode)
  }
  return requestJson<MemoryDeleteOperationListPayload>(`/delete/operations?${params.toString()}`)
}

export async function getMemoryDeleteOperation(
  operationId: string,
): Promise<MemoryDeleteOperationDetailPayload> {
  return requestJson<MemoryDeleteOperationDetailPayload>(`/delete/operations/${encodeURIComponent(operationId)}`)
}

export async function getMemorySources(): Promise<MemorySourceListPayload> {
  return requestJson<MemorySourceListPayload>('/sources')
}

export async function getMemoryRuntimeConfig(): Promise<MemoryRuntimeConfigPayload> {
  return requestJson<MemoryRuntimeConfigPayload>('/runtime/config')
}

export async function refreshMemoryRuntimeSelfCheck(): Promise<MemoryRuntimeSelfCheckPayload> {
  return requestJson<MemoryRuntimeSelfCheckPayload>('/runtime/self-check/refresh', {
    method: 'POST',
  })
}

export async function getMemoryConfigSchema(): Promise<MemoryConfigSchemaPayload> {
  return requestJson<MemoryConfigSchemaPayload>('/config/schema')
}

export async function getMemoryConfig(): Promise<MemoryConfigPayload> {
  return requestJson<MemoryConfigPayload>('/config')
}

export async function updateMemoryConfig(config: Record<string, unknown>): Promise<{ success: boolean; message?: string }> {
  return requestJson('/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  })
}

export async function getMemoryConfigRaw(): Promise<MemoryRawConfigPayload> {
  return requestJson<MemoryRawConfigPayload>('/config/raw')
}

export async function updateMemoryConfigRaw(config: string): Promise<{ success: boolean; message?: string }> {
  return requestJson('/config/raw', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  })
}

export async function getMemoryImportGuide(): Promise<MemoryImportGuidePayload> {
  return requestJson<MemoryImportGuidePayload>('/import/guide')
}

export async function getMemoryImportSettings(): Promise<Record<string, unknown>> {
  return requestJson('/import/settings')
}

export async function getMemoryImportTasks(limit: number = 20): Promise<MemoryTaskListPayload> {
  return requestJson<MemoryTaskListPayload>(`/import/tasks?limit=${limit}`)
}

export async function createMemoryPasteImport(payload: Record<string, unknown>): Promise<{ success: boolean; task?: MemoryTaskPayload }> {
  return requestJson('/import/paste', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function getMemoryTuningProfile(): Promise<MemoryTuningProfilePayload> {
  return requestJson<MemoryTuningProfilePayload>('/retrieval_tuning/profile')
}

export async function getMemoryTuningTasks(limit: number = 20): Promise<MemoryTaskListPayload> {
  return requestJson<MemoryTaskListPayload>(`/retrieval_tuning/tasks?limit=${limit}`)
}

export async function createMemoryTuningTask(payload: Record<string, unknown>): Promise<{ success: boolean; task?: MemoryTaskPayload }> {
  return requestJson('/retrieval_tuning/tasks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function applyBestMemoryTuningProfile(taskId: string): Promise<{ success: boolean; error?: string }> {
  return requestJson(`/retrieval_tuning/tasks/${encodeURIComponent(taskId)}/apply-best`, {
    method: 'POST',
  })
}

export async function getMemoryTuningReport(taskId: string, format: 'md' | 'json' = 'md'): Promise<{ success: boolean; content: string; path: string; error?: string }> {
  return requestJson(`/retrieval_tuning/tasks/${encodeURIComponent(taskId)}/report?format=${format}`)
}
