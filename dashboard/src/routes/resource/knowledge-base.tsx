import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'

import {
  Database,
  FileDown,
  Gauge,
  RefreshCw,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react'

import { CodeEditor, MarkdownRenderer } from '@/components'
import { MemoryDeleteDialog } from '@/components/memory/MemoryDeleteDialog'
import { MemoryConfigEditor } from '@/components/memory/MemoryConfigEditor'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { useToast } from '@/hooks/use-toast'
import { cn } from '@/lib/utils'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  executeMemoryDelete,
  applyBestMemoryTuningProfile,
  createMemoryPasteImport,
  createMemoryTuningTask,
  getMemoryConfig,
  getMemoryConfigRaw,
  getMemoryConfigSchema,
  getMemoryDeleteOperation,
  getMemoryDeleteOperations,
  getMemoryImportGuide,
  getMemoryImportTasks,
  getMemoryRuntimeConfig,
  getMemorySources,
  getMemoryTuningProfile,
  getMemoryTuningTasks,
  type MemoryDeleteRequestPayload,
  previewMemoryDelete,
  refreshMemoryRuntimeSelfCheck,
  restoreMemoryDelete,
  updateMemoryConfig,
  updateMemoryConfigRaw,
  type MemoryConfigSchemaPayload,
  type MemoryDeleteExecutePayload,
  type MemoryDeleteOperationPayload,
  type MemorySourceItemPayload,
  type MemoryRuntimeConfigPayload,
  type MemoryTaskPayload,
} from '@/lib/memory-api'

const DELETE_OPERATION_FETCH_LIMIT = 100
const DELETE_OPERATION_PAGE_SIZE = 6
const DELETE_OPERATION_ITEM_PAGE_SIZE = 8

function formatDeleteOperationMode(mode: string): string {
  switch (mode) {
    case 'entity':
      return '实体'
    case 'relation':
      return '关系'
    case 'paragraph':
      return '段落'
    case 'source':
      return '来源'
    case 'mixed':
      return '混合'
    default:
      return mode || '未知'
  }
}

function formatDeleteOperationStatus(status: string): string {
  switch (status) {
    case 'executed':
      return '已执行'
    case 'restored':
      return '已恢复'
    default:
      return status || '未知'
  }
}

function formatDeleteOperationTime(timestamp?: number | null): string {
  if (!timestamp) {
    return '未知时间'
  }
  const normalized = timestamp > 1_000_000_000_000 ? timestamp : timestamp * 1000
  const value = new Date(normalized)
  if (Number.isNaN(value.getTime())) {
    return '未知时间'
  }
  return value.toLocaleString('zh-CN', {
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

type DeleteOperationItem = NonNullable<MemoryDeleteOperationPayload['items']>[number]

function trimDeleteItemText(value: string, maxLength: number = 140): string {
  const normalized = String(value ?? '').trim().replace(/\s+/g, ' ')
  if (!normalized) {
    return ''
  }
  if (normalized.length <= maxLength) {
    return normalized
  }
  return `${normalized.slice(0, maxLength)}...`
}

function formatDeleteRelationText(subject: string, predicate: string, object: string): string {
  const left = String(subject ?? '').trim()
  const middle = String(predicate ?? '').trim()
  const right = String(object ?? '').trim()
  return [left, middle, right].filter(Boolean).join(' -> ')
}

function getDeleteOperationItemLabel(item: DeleteOperationItem): string {
  const payload = item.payload ?? {}
  if (item.item_type === 'entity') {
    const entity = (payload.entity ?? {}) as Record<string, unknown>
    return String(entity.name ?? item.item_key ?? item.item_hash ?? '未命名实体')
  }
  if (item.item_type === 'relation') {
    const relation = (payload.relation ?? {}) as Record<string, unknown>
    return (
      formatDeleteRelationText(
        String(relation.subject ?? ''),
        String(relation.predicate ?? ''),
        String(relation.object ?? ''),
      ) || String(item.item_key ?? item.item_hash ?? '未命名关系')
    )
  }
  if (item.item_type === 'paragraph') {
    const paragraph = (payload.paragraph ?? {}) as Record<string, unknown>
    const source = String(paragraph.source ?? '').trim()
    return source || String(item.item_key ?? item.item_hash ?? '未命名段落')
  }
  return String(item.item_key ?? item.item_hash ?? '未命名对象')
}

function getDeleteOperationItemPreview(item: DeleteOperationItem): string {
  const payload = item.payload ?? {}
  if (item.item_type === 'entity') {
    const paragraphLinks = Array.isArray(payload.paragraph_links) ? payload.paragraph_links : []
    if (paragraphLinks.length > 0) {
      return `关联段落 ${paragraphLinks.length} 个`
    }
    return '实体快照'
  }
  if (item.item_type === 'relation') {
    const relation = (payload.relation ?? {}) as Record<string, unknown>
    const paragraphHashes = Array.isArray(payload.paragraph_hashes) ? payload.paragraph_hashes : []
    const confidence = relation.confidence
    const parts = []
    if (paragraphHashes.length > 0) {
      parts.push(`证据段落 ${paragraphHashes.length} 个`)
    }
    if (typeof confidence === 'number') {
      parts.push(`置信度 ${confidence.toFixed(2)}`)
    }
    return parts.join('，') || '关系快照'
  }
  if (item.item_type === 'paragraph') {
    const paragraph = (payload.paragraph ?? {}) as Record<string, unknown>
    return trimDeleteItemText(String(paragraph.content ?? ''))
  }
  return ''
}

function getDeleteOperationItemSource(item: DeleteOperationItem): string {
  const payload = item.payload ?? {}
  if (item.item_type === 'paragraph') {
    const paragraph = (payload.paragraph ?? {}) as Record<string, unknown>
    return String(paragraph.source ?? '').trim()
  }
  return String(payload.source ?? '').trim()
}

export function KnowledgeBasePage() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [refreshingCheck, setRefreshingCheck] = useState(false)
  const [creatingImport, setCreatingImport] = useState(false)
  const [creatingTuning, setCreatingTuning] = useState(false)
  const [rawMode, setRawMode] = useState(false)

  const [schemaPayload, setSchemaPayload] = useState<MemoryConfigSchemaPayload | null>(null)
  const [visualConfig, setVisualConfig] = useState<Record<string, unknown>>({})
  const [rawConfig, setRawConfig] = useState('')
  const [runtimeConfig, setRuntimeConfig] = useState<MemoryRuntimeConfigPayload | null>(null)
  const [selfCheckReport, setSelfCheckReport] = useState<Record<string, unknown> | null>(null)
  const [importGuide, setImportGuide] = useState('')
  const [importTasks, setImportTasks] = useState<MemoryTaskPayload[]>([])
  const [tuningTasks, setTuningTasks] = useState<MemoryTaskPayload[]>([])
  const [tuningProfile, setTuningProfile] = useState<Record<string, unknown>>({})
  const [tuningProfileToml, setTuningProfileToml] = useState('')
  const [memorySources, setMemorySources] = useState<MemorySourceItemPayload[]>([])
  const [deleteOperations, setDeleteOperations] = useState<MemoryDeleteOperationPayload[]>([])
  const [selectedOperationDetail, setSelectedOperationDetail] = useState<MemoryDeleteOperationPayload | null>(null)
  const [selectedOperationDetailLoading, setSelectedOperationDetailLoading] = useState(false)
  const [selectedOperationDetailError, setSelectedOperationDetailError] = useState('')
  const [sourceSearch, setSourceSearch] = useState('')
  const [operationSearch, setOperationSearch] = useState('')
  const [operationModeFilter, setOperationModeFilter] = useState('all')
  const [operationStatusFilter, setOperationStatusFilter] = useState('all')
  const [operationPage, setOperationPage] = useState(1)
  const [selectedOperationId, setSelectedOperationId] = useState('')
  const [selectedOperationItemSearch, setSelectedOperationItemSearch] = useState('')
  const [selectedOperationItemPage, setSelectedOperationItemPage] = useState(1)
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [deleteDialogTitle, setDeleteDialogTitle] = useState('删除预览')
  const [deleteDialogDescription, setDeleteDialogDescription] = useState('')
  const [deletePreview, setDeletePreview] = useState<Awaited<ReturnType<typeof previewMemoryDelete>> | null>(null)
  const [deletePreviewError, setDeletePreviewError] = useState<string | null>(null)
  const [deletePreviewLoading, setDeletePreviewLoading] = useState(false)
  const [deleteExecuting, setDeleteExecuting] = useState(false)
  const [deleteRestoring, setDeleteRestoring] = useState(false)
  const [deleteResult, setDeleteResult] = useState<MemoryDeleteExecutePayload | null>(null)
  const [pendingDeleteRequest, setPendingDeleteRequest] = useState<MemoryDeleteRequestPayload | null>(null)

  const [pasteName, setPasteName] = useState('')
  const [pasteMode, setPasteMode] = useState('text')
  const [pasteContent, setPasteContent] = useState('')
  const [tuningObjective, setTuningObjective] = useState('precision_priority')
  const [tuningIntensity, setTuningIntensity] = useState('standard')
  const [tuningSampleSize, setTuningSampleSize] = useState('24')
  const [tuningTopKEval, setTuningTopKEval] = useState('20')

  const loadPage = useCallback(async () => {
    try {
      setLoading(true)
      const [
        schema,
        configPayload,
        rawPayload,
        runtimePayload,
        guidePayload,
        importTaskPayload,
        tuningProfilePayload,
        tuningTaskPayload,
        sourcePayload,
        deleteOperationPayload,
      ] = await Promise.all([
        getMemoryConfigSchema(),
        getMemoryConfig(),
        getMemoryConfigRaw(),
        getMemoryRuntimeConfig(),
        getMemoryImportGuide(),
        getMemoryImportTasks(20),
        getMemoryTuningProfile(),
        getMemoryTuningTasks(20),
        getMemorySources(),
        getMemoryDeleteOperations(DELETE_OPERATION_FETCH_LIMIT),
      ])

      setSchemaPayload(schema)
      setVisualConfig(configPayload.config ?? {})
      setRawConfig(rawPayload.config ?? '')
      setRuntimeConfig(runtimePayload)
      setImportGuide(guidePayload.content ?? '')
      setImportTasks(importTaskPayload.items ?? [])
      setTuningProfile(tuningProfilePayload.profile ?? {})
      setTuningProfileToml(tuningProfilePayload.toml ?? '')
      setTuningTasks(tuningTaskPayload.items ?? [])
      setMemorySources(sourcePayload.items ?? [])
      setDeleteOperations(deleteOperationPayload.items ?? [])
    } catch (error) {
      toast({
        title: '加载长期记忆控制台失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => {
    void loadPage()
  }, [loadPage])

  const configPath = schemaPayload?.path ?? 'config/a_memorix.toml'
  const schema = schemaPayload?.schema

  const runtimeBadges = useMemo(() => {
    if (!runtimeConfig) {
      return []
    }
    return [
      { label: '运行状态', value: runtimeConfig.runtime_ready ? '就绪' : '未就绪' },
      { label: 'Embedding 维度', value: String(runtimeConfig.embedding_dimension) },
      { label: '自动保存', value: runtimeConfig.auto_save ? '开启' : '关闭' },
      { label: '数据目录', value: runtimeConfig.data_dir },
    ]
  }, [runtimeConfig])

  const filteredSources = useMemo(() => {
    const keyword = sourceSearch.trim().toLowerCase()
    if (!keyword) {
      return memorySources
    }
    return memorySources.filter((item) => String(item.source ?? '').toLowerCase().includes(keyword))
  }, [memorySources, sourceSearch])

  const filteredDeleteOperations = useMemo(() => {
    const keyword = operationSearch.trim().toLowerCase()
    return deleteOperations.filter((operation) => {
      const mode = String(operation.mode ?? '').trim()
      const status = String(operation.status ?? '').trim()
      const summary = operation.summary ?? {}
      const sources = Array.isArray(summary.sources) ? summary.sources : []

      if (operationModeFilter !== 'all' && mode !== operationModeFilter) {
        return false
      }
      if (operationStatusFilter !== 'all' && status !== operationStatusFilter) {
        return false
      }
      if (!keyword) {
        return true
      }

      return [
        operation.operation_id,
        operation.reason,
        operation.requested_by,
        mode,
        status,
        ...sources.map((item) => String(item)),
      ]
        .map((item) => String(item ?? '').toLowerCase())
        .some((item) => item.includes(keyword))
    })
  }, [deleteOperations, operationModeFilter, operationSearch, operationStatusFilter])

  const deleteOperationPageCount = Math.max(1, Math.ceil(filteredDeleteOperations.length / DELETE_OPERATION_PAGE_SIZE))
  const pagedDeleteOperations = useMemo(() => {
    const start = (operationPage - 1) * DELETE_OPERATION_PAGE_SIZE
    return filteredDeleteOperations.slice(start, start + DELETE_OPERATION_PAGE_SIZE)
  }, [filteredDeleteOperations, operationPage])

  const selectedDeleteOperation = useMemo(
    () => filteredDeleteOperations.find((operation) => operation.operation_id === selectedOperationId) ?? pagedDeleteOperations[0] ?? null,
    [filteredDeleteOperations, pagedDeleteOperations, selectedOperationId],
  )

  useEffect(() => {
    setOperationPage(1)
  }, [operationSearch, operationModeFilter, operationStatusFilter])

  useEffect(() => {
    if (operationPage > deleteOperationPageCount) {
      setOperationPage(deleteOperationPageCount)
    }
  }, [deleteOperationPageCount, operationPage])

  useEffect(() => {
    if (!selectedDeleteOperation) {
      if (selectedOperationId) {
        setSelectedOperationId('')
      }
      setSelectedOperationDetail(null)
      setSelectedOperationDetailError('')
      return
    }
    if (selectedDeleteOperation.operation_id !== selectedOperationId) {
      setSelectedOperationId(selectedDeleteOperation.operation_id)
    }
  }, [selectedDeleteOperation, selectedOperationId])

  useEffect(() => {
    const operationId = selectedDeleteOperation?.operation_id
    if (!operationId) {
      setSelectedOperationDetail(null)
      setSelectedOperationDetailError('')
      return
    }

    let cancelled = false
    setSelectedOperationDetailLoading(true)
    setSelectedOperationDetailError('')

    void getMemoryDeleteOperation(operationId)
      .then((payload) => {
        if (cancelled) {
          return
        }
        if (!payload.success || !payload.operation) {
          setSelectedOperationDetail(null)
          setSelectedOperationDetailError(payload.error || '未能加载删除操作详情')
          return
        }
        setSelectedOperationDetail(payload.operation)
      })
      .catch((error) => {
        if (cancelled) {
          return
        }
        setSelectedOperationDetail(null)
        setSelectedOperationDetailError(error instanceof Error ? error.message : '未能加载删除操作详情')
      })
      .finally(() => {
        if (!cancelled) {
          setSelectedOperationDetailLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [selectedDeleteOperation?.operation_id])

  const toggleSourceSelection = useCallback((source: string, checked: boolean) => {
    setSelectedSources((current) => {
      if (checked) {
        return current.includes(source) ? current : [...current, source]
      }
      return current.filter((item) => item !== source)
    })
  }, [])

  const openSourceDeletePreview = useCallback(async () => {
    if (selectedSources.length <= 0) {
      toast({
        title: '请选择来源',
        description: '至少选择一个来源后再进行删除预览。',
        variant: 'destructive',
      })
      return
    }
    const request: MemoryDeleteRequestPayload = {
      mode: 'source',
      selector: { sources: selectedSources },
      reason: 'knowledge_base_source_delete',
      requested_by: 'knowledge_base',
    }
    setDeleteDialogTitle('批量删除来源')
    setDeleteDialogDescription('删除来源只会删除该来源下的段落，以及失去全部证据的关系，不会自动删除实体。')
    setPendingDeleteRequest(request)
    setDeletePreview(null)
    setDeleteResult(null)
    setDeletePreviewError(null)
    setDeleteDialogOpen(true)
    setDeletePreviewLoading(true)
    try {
      const preview = await previewMemoryDelete(request)
      setDeletePreview(preview)
    } catch (error) {
      setDeletePreviewError(error instanceof Error ? error.message : '删除预览失败')
    } finally {
      setDeletePreviewLoading(false)
    }
  }, [selectedSources, toast])

  const executePendingDelete = useCallback(async () => {
    if (!pendingDeleteRequest) {
      return
    }
    try {
      setDeleteExecuting(true)
      const result = await executeMemoryDelete(pendingDeleteRequest)
      setDeleteResult(result)
      toast({
        title: result.success ? '删除成功' : '删除失败',
        description: result.success ? `操作 ${result.operation_id} 已完成` : result.error || '未能执行删除',
        variant: result.success ? 'default' : 'destructive',
      })
      if (result.success) {
        const [sourcePayload, deleteOperationPayload] = await Promise.all([
          getMemorySources(),
          getMemoryDeleteOperations(DELETE_OPERATION_FETCH_LIMIT),
        ])
        setMemorySources(sourcePayload.items ?? [])
        setDeleteOperations(deleteOperationPayload.items ?? [])
        setSelectedSources([])
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
  }, [pendingDeleteRequest, toast])

  const restoreDeleteOperation = useCallback(async (operationId: string) => {
    try {
      setDeleteRestoring(true)
      await restoreMemoryDelete({ operation_id: operationId, requested_by: 'knowledge_base' })
      toast({
        title: '恢复成功',
        description: `删除操作 ${operationId} 已恢复`,
      })
      setDeleteDialogOpen(false)
      const [sourcePayload, deleteOperationPayload] = await Promise.all([
        getMemorySources(),
        getMemoryDeleteOperations(DELETE_OPERATION_FETCH_LIMIT),
      ])
      setMemorySources(sourcePayload.items ?? [])
      setDeleteOperations(deleteOperationPayload.items ?? [])
    } catch (error) {
      toast({
        title: '恢复失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setDeleteRestoring(false)
    }
  }, [toast])

  const closeDeleteDialog = useCallback((open: boolean) => {
    if (!open) {
      setDeleteDialogOpen(false)
      setDeletePreview(null)
      setDeleteResult(null)
      setDeletePreviewError(null)
      setPendingDeleteRequest(null)
      return
    }
    setDeleteDialogOpen(true)
  }, [])

  const selectedOperationResolved = useMemo(() => {
    if (!selectedDeleteOperation) {
      return null
    }
    if (selectedOperationDetail?.operation_id === selectedDeleteOperation.operation_id) {
      return {
        ...selectedDeleteOperation,
        ...selectedOperationDetail,
      } satisfies MemoryDeleteOperationPayload
    }
    return selectedDeleteOperation
  }, [selectedDeleteOperation, selectedOperationDetail])
  const selectedOperationSummaryResolved = ((selectedOperationResolved?.summary ?? {}) as Record<string, unknown>)
  const selectedOperationCounts = ((selectedOperationSummaryResolved.counts as Record<string, number> | undefined) ?? {})
  const selectedOperationSources = Array.isArray(selectedOperationSummaryResolved.sources)
    ? selectedOperationSummaryResolved.sources.map((item) => String(item)).filter(Boolean)
    : []
  const selectedOperationItems = Array.isArray(selectedOperationResolved?.items)
    ? selectedOperationResolved.items
    : []
  const filteredSelectedOperationItems = useMemo(() => {
    const keyword = selectedOperationItemSearch.trim().toLowerCase()
    if (!keyword) {
      return selectedOperationItems
    }
    return selectedOperationItems.filter((item) => {
      const payload = item.payload ?? {}
      const source = String(payload.source ?? '').trim()
      return [
        item.item_type,
        item.item_hash,
        item.item_key,
        source,
      ]
        .map((value) => String(value ?? '').toLowerCase())
        .some((value) => value.includes(keyword))
    })
  }, [selectedOperationItemSearch, selectedOperationItems])
  const selectedOperationItemPageCount = Math.max(
    1,
    Math.ceil(filteredSelectedOperationItems.length / DELETE_OPERATION_ITEM_PAGE_SIZE),
  )
  const pagedSelectedOperationItems = useMemo(() => {
    const start = (selectedOperationItemPage - 1) * DELETE_OPERATION_ITEM_PAGE_SIZE
    return filteredSelectedOperationItems.slice(start, start + DELETE_OPERATION_ITEM_PAGE_SIZE)
  }, [filteredSelectedOperationItems, selectedOperationItemPage])

  useEffect(() => {
    setSelectedOperationItemPage(1)
  }, [selectedOperationId, selectedOperationItemSearch])

  useEffect(() => {
    if (selectedOperationItemPage > selectedOperationItemPageCount) {
      setSelectedOperationItemPage(selectedOperationItemPageCount)
    }
  }, [selectedOperationItemPage, selectedOperationItemPageCount])

  const saveVisualConfig = useCallback(async () => {
    try {
      setSaving(true)
      await updateMemoryConfig(visualConfig)
      const [nextConfig, nextRaw, nextRuntime] = await Promise.all([
        getMemoryConfig(),
        getMemoryConfigRaw(),
        getMemoryRuntimeConfig(),
      ])
      setVisualConfig(nextConfig.config)
      setRawConfig(nextRaw.config)
      setRuntimeConfig(nextRuntime)
      toast({ title: '配置已保存', description: '长期记忆配置已经应用到运行时。' })
    } catch (error) {
      toast({
        title: '保存配置失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setSaving(false)
    }
  }, [toast, visualConfig])

  const saveRaw = useCallback(async () => {
    try {
      setSaving(true)
      await updateMemoryConfigRaw(rawConfig)
      const [nextConfig, nextRuntime] = await Promise.all([getMemoryConfig(), getMemoryRuntimeConfig()])
      setVisualConfig(nextConfig.config)
      setRuntimeConfig(nextRuntime)
      toast({ title: '原始 TOML 已保存', description: '长期记忆配置已经重新加载。' })
    } catch (error) {
      toast({
        title: '保存原始配置失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setSaving(false)
    }
  }, [rawConfig, toast])

  const refreshSelfCheck = useCallback(async () => {
    try {
      setRefreshingCheck(true)
      const payload = await refreshMemoryRuntimeSelfCheck()
      setSelfCheckReport((payload.report ?? null) as Record<string, unknown> | null)
      const nextRuntime = await getMemoryRuntimeConfig()
      setRuntimeConfig(nextRuntime)
      toast({
        title: payload.success ? '自检通过' : '自检未通过',
        description: payload.success ? '运行时状态正常。' : '请检查 embedding 配置和外部服务连通性。',
        variant: payload.success ? 'default' : 'destructive',
      })
    } catch (error) {
      toast({
        title: '运行时自检失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setRefreshingCheck(false)
    }
  }, [toast])

  const submitPasteImport = useCallback(async () => {
    try {
      setCreatingImport(true)
      await createMemoryPasteImport({
        name: pasteName || undefined,
        content: pasteContent,
        input_mode: pasteMode,
      })
      const tasks = await getMemoryImportTasks(20)
      setImportTasks(tasks.items ?? [])
      setPasteContent('')
      setPasteName('')
      toast({ title: '导入任务已创建', description: '粘贴内容已经进入 A_Memorix 导入队列。' })
    } catch (error) {
      toast({
        title: '创建导入任务失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setCreatingImport(false)
    }
  }, [pasteContent, pasteMode, pasteName, toast])

  const submitTuningTask = useCallback(async () => {
    try {
      setCreatingTuning(true)
      await createMemoryTuningTask({
        objective: tuningObjective,
        intensity: tuningIntensity,
        sample_size: Number(tuningSampleSize),
        top_k_eval: Number(tuningTopKEval),
      })
      const tasks = await getMemoryTuningTasks(20)
      setTuningTasks(tasks.items ?? [])
      toast({ title: '调优任务已创建', description: '新的检索调优任务已经进入队列。' })
    } catch (error) {
      toast({
        title: '创建调优任务失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    } finally {
      setCreatingTuning(false)
    }
  }, [toast, tuningIntensity, tuningObjective, tuningSampleSize, tuningTopKEval])

  const applyBestTask = useCallback(async (taskId: string) => {
    try {
      await applyBestMemoryTuningProfile(taskId)
      const [profilePayload, runtimePayload, tuningTaskPayload] = await Promise.all([
        getMemoryTuningProfile(),
        getMemoryRuntimeConfig(),
        getMemoryTuningTasks(20),
      ])
      setTuningProfile(profilePayload.profile ?? {})
      setTuningProfileToml(profilePayload.toml ?? '')
      setRuntimeConfig(runtimePayload)
      setTuningTasks(tuningTaskPayload.items ?? [])
      toast({ title: '最佳参数已应用', description: `任务 ${taskId} 的最佳轮次已经写入运行时。` })
    } catch (error) {
      toast({
        title: '应用最佳参数失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    }
  }, [toast])

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="rounded-xl border bg-background px-6 py-5 text-sm text-muted-foreground shadow-sm">
          正在加载长期记忆控制台...
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none border-b bg-card/60 px-6 py-4 backdrop-blur">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-2xl font-bold">长期记忆控制台</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              统一管理 A_Memorix 的配置、自检、导入和检索调优，替代旧 LPMM 知识库管理入口。
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => navigate({ to: '/resource/knowledge-graph' })}>
              <Database className="mr-2 h-4 w-4" />
              打开图谱
            </Button>
            <Button variant="outline" onClick={() => void loadPage()}>
              <RefreshCw className="mr-2 h-4 w-4" />
              刷新数据
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6">
        <div className="mx-auto flex max-w-7xl flex-col gap-6">
          <div className="grid gap-4 xl:grid-cols-4">
            {runtimeBadges.map((item) => (
              <Card key={item.label}>
                <CardHeader className="pb-2">
                  <CardDescription>{item.label}</CardDescription>
                  <CardTitle className="break-all text-base">{item.value}</CardTitle>
                </CardHeader>
              </Card>
            ))}
          </div>

          <Tabs defaultValue="overview" className="space-y-4">
            <TabsList className="h-auto flex-wrap justify-start">
              <TabsTrigger value="overview">概览</TabsTrigger>
              <TabsTrigger value="config">配置</TabsTrigger>
              <TabsTrigger value="import">导入</TabsTrigger>
              <TabsTrigger value="tuning">调优</TabsTrigger>
              <TabsTrigger value="delete">删除</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
                <Card>
                  <CardHeader className="flex flex-row items-start justify-between space-y-0">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Gauge className="h-4 w-4" />
                        运行时自检
                      </CardTitle>
                      <CardDescription>用于确认 embedding、向量库与运行时状态是否一致。</CardDescription>
                    </div>
                    <Button size="sm" onClick={() => void refreshSelfCheck()} disabled={refreshingCheck}>
                      <RefreshCw className={`mr-2 h-4 w-4 ${refreshingCheck ? 'animate-spin' : ''}`} />
                      重新自检
                    </Button>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Alert>
                      <AlertDescription>
                        当前配置文件路径：<code>{configPath}</code>
                      </AlertDescription>
                    </Alert>
                    <CodeEditor
                      value={JSON.stringify(selfCheckReport ?? runtimeConfig ?? {}, null, 2)}
                      language="json"
                      readOnly
                      height="320px"
                    />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4" />
                      当前运行态摘要
                    </CardTitle>
                    <CardDescription>这里展示运行态重点指标，方便先判断是否需要导入或调优。</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4 text-sm">
                    <div className="flex flex-wrap gap-2">
                      <Badge variant={runtimeConfig?.runtime_ready ? 'default' : 'destructive'}>
                        {runtimeConfig?.runtime_ready ? '运行就绪' : '运行未就绪'}
                      </Badge>
                      <Badge variant={runtimeConfig?.embedding_degraded ? 'destructive' : 'secondary'}>
                        {runtimeConfig?.embedding_degraded ? 'Embedding 已退化' : 'Embedding 正常'}
                      </Badge>
                    </div>
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="rounded-lg border bg-muted/30 p-3">
                        <div className="text-xs text-muted-foreground">待补回段落向量</div>
                        <div className="mt-1 text-2xl font-semibold">{runtimeConfig?.paragraph_vector_backfill_pending ?? 0}</div>
                      </div>
                      <div className="rounded-lg border bg-muted/30 p-3">
                        <div className="text-xs text-muted-foreground">失败补回任务</div>
                        <div className="mt-1 text-2xl font-semibold">{runtimeConfig?.paragraph_vector_backfill_failed ?? 0}</div>
                      </div>
                    </div>
                    <div className="rounded-lg border bg-muted/30 p-3">
                      <div className="text-xs text-muted-foreground">当前调优配置</div>
                      <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words text-xs">
                        {JSON.stringify(tuningProfile, null, 2)}
                      </pre>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="config" className="space-y-4">
              <Card>
                <CardHeader className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <SlidersHorizontal className="h-4 w-4" />
                      长期记忆配置
                    </CardTitle>
                    <CardDescription>
                      常用字段可视化编辑，长尾高级项继续通过原始 TOML 维护。
                    </CardDescription>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button variant={rawMode ? 'outline' : 'default'} onClick={() => setRawMode(false)}>
                      可视化配置
                    </Button>
                    <Button variant={rawMode ? 'default' : 'outline'} onClick={() => setRawMode(true)}>
                      原始 TOML
                    </Button>
                    <Button onClick={() => void (rawMode ? saveRaw() : saveVisualConfig())} disabled={saving}>
                      <Save className="mr-2 h-4 w-4" />
                      保存
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Alert>
                    <AlertDescription>
                      当前配置文件：<code>{configPath}</code>
                      {schema?._note ? `；${schema._note}` : ''}
                    </AlertDescription>
                  </Alert>

                  {rawMode ? (
                    <CodeEditor
                      value={rawConfig}
                      onChange={setRawConfig}
                      language="toml"
                      height="620px"
                    />
                  ) : schema ? (
                    <MemoryConfigEditor
                      schema={schema}
                      config={visualConfig}
                      onChange={setVisualConfig}
                      disabled={saving}
                    />
                  ) : (
                    <div className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
                      当前未能加载配置 schema，请先刷新页面或检查后端日志。
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="import" className="space-y-4">
              <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Upload className="h-4 w-4" />
                      粘贴导入
                    </CardTitle>
                    <CardDescription>先提供一个主线内可用的轻量导入入口，避免用户只能回到旧静态页。</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label>名称</Label>
                        <Input value={pasteName} onChange={(event) => setPasteName(event.target.value)} placeholder="可选：如 private_alice_weekend.txt" />
                      </div>
                      <div className="space-y-2">
                        <Label>输入模式</Label>
                        <Select value={pasteMode} onValueChange={setPasteMode}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="text">text</SelectItem>
                            <SelectItem value="json">json</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label>粘贴内容</Label>
                      <Textarea
                        value={pasteContent}
                        onChange={(event) => setPasteContent(event.target.value)}
                        rows={12}
                        placeholder="将聊天整理结果、文档片段或结构化 JSON 直接粘贴到这里。"
                      />
                    </div>
                    <Button onClick={() => void submitPasteImport()} disabled={creatingImport || !pasteContent.trim()}>
                      <Upload className="mr-2 h-4 w-4" />
                      创建导入任务
                    </Button>
                  </CardContent>
                </Card>

                <div className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <FileDown className="h-4 w-4" />
                        导入指南
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <MarkdownRenderer content={importGuide || '暂无导入指南内容。'} />
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>最近导入任务</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>任务</TableHead>
                            <TableHead>状态</TableHead>
                            <TableHead>模式</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {importTasks.length > 0 ? importTasks.map((task) => (
                            <TableRow key={String(task.task_id ?? Math.random())}>
                              <TableCell className="font-mono text-xs">{String(task.task_id ?? '-')}</TableCell>
                              <TableCell>{String(task.status ?? '-')}</TableCell>
                              <TableCell>{String(task.mode ?? task.source ?? '-')}</TableCell>
                            </TableRow>
                          )) : (
                            <TableRow>
                              <TableCell colSpan={3} className="text-center text-muted-foreground">
                                还没有导入任务
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="tuning" className="space-y-4">
              <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4" />
                      调优任务
                    </CardTitle>
                    <CardDescription>先把创建、查看、应用最佳结果这条调优闭环接到主线控制台。</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label>目标函数</Label>
                        <Select value={tuningObjective} onValueChange={setTuningObjective}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="precision_priority">precision_priority</SelectItem>
                            <SelectItem value="balanced">balanced</SelectItem>
                            <SelectItem value="recall_priority">recall_priority</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label>强度</Label>
                        <Select value={tuningIntensity} onValueChange={setTuningIntensity}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="quick">quick</SelectItem>
                            <SelectItem value="standard">standard</SelectItem>
                            <SelectItem value="deep">deep</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label>样本量</Label>
                        <Input type="number" value={tuningSampleSize} onChange={(event) => setTuningSampleSize(event.target.value)} />
                      </div>
                      <div className="space-y-2">
                        <Label>评估 Top-K</Label>
                        <Input type="number" value={tuningTopKEval} onChange={(event) => setTuningTopKEval(event.target.value)} />
                      </div>
                    </div>
                    <Button onClick={() => void submitTuningTask()} disabled={creatingTuning}>
                      <Sparkles className="mr-2 h-4 w-4" />
                      创建调优任务
                    </Button>
                  </CardContent>
                </Card>

                <div className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>当前调优配置快照</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <CodeEditor
                        value={JSON.stringify(tuningProfile, null, 2)}
                        language="json"
                        readOnly
                        height="220px"
                      />
                      <CodeEditor
                        value={tuningProfileToml}
                        language="toml"
                        readOnly
                        height="180px"
                      />
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>最近调优任务</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>任务</TableHead>
                            <TableHead>状态</TableHead>
                            <TableHead>动作</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {tuningTasks.length > 0 ? tuningTasks.map((task) => (
                            <TableRow key={String(task.task_id ?? Math.random())}>
                              <TableCell className="font-mono text-xs">{String(task.task_id ?? '-')}</TableCell>
                              <TableCell>{String(task.status ?? '-')}</TableCell>
                              <TableCell>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => void applyBestTask(String(task.task_id ?? ''))}
                                  disabled={!task.task_id}
                                >
                                  应用最佳
                                </Button>
                              </TableCell>
                            </TableRow>
                          )) : (
                            <TableRow>
                              <TableCell colSpan={3} className="text-center text-muted-foreground">
                                还没有调优任务
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="delete" className="space-y-4">
              <div className="space-y-4">
                <Card>
                  <CardHeader className="space-y-3">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Trash2 className="h-4 w-4" />
                        来源批量删除
                      </CardTitle>
                      <CardDescription>
                        用于按来源批量清理测试数据或指定导入批次。不会自动删除实体，只会删除来源段落和失去全部证据的关系。
                      </CardDescription>
                    </div>
                    <Alert>
                      <AlertDescription>
                        建议先在图谱里确认影响范围，再在这里做批量来源删除。所有删除都会先经过预览，并支持按 operation 恢复。
                      </AlertDescription>
                    </Alert>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
                      <div className="space-y-2">
                        <Label>来源检索</Label>
                        <Input
                          value={sourceSearch}
                          onChange={(event) => setSourceSearch(event.target.value)}
                          placeholder="搜索 source 名称"
                        />
                      </div>
                      <div className="flex flex-wrap gap-2 lg:justify-end">
                        <Button
                          variant="outline"
                          onClick={() => setSelectedSources(filteredSources.map((item) => String(item.source ?? '')).filter(Boolean))}
                        >
                          全选当前结果
                        </Button>
                        <Button onClick={() => void openSourceDeletePreview()} disabled={selectedSources.length <= 0}>
                          <Trash2 className="mr-2 h-4 w-4" />
                          预览删除
                        </Button>
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                      <Badge variant="outline">当前命中 {filteredSources.length} 个来源</Badge>
                      <Badge variant="secondary">已选择 {selectedSources.length} 个来源</Badge>
                    </div>

                    <ScrollArea className="h-[320px] rounded-lg border">
                      <Table>
                        <TableHeader className="sticky top-0 bg-background">
                          <TableRow>
                            <TableHead className="w-12">选中</TableHead>
                            <TableHead>来源</TableHead>
                            <TableHead>段落数</TableHead>
                            <TableHead>关系数</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {filteredSources.length > 0 ? filteredSources.map((item) => {
                            const source = String(item.source ?? '')
                            const checked = selectedSources.includes(source)
                            return (
                              <TableRow key={source}>
                                <TableCell>
                                  <Checkbox checked={checked} onCheckedChange={(value) => toggleSourceSelection(source, Boolean(value))} />
                                </TableCell>
                                <TableCell className="font-mono text-xs break-all">{source}</TableCell>
                                <TableCell>{Number(item.paragraph_count ?? 0)}</TableCell>
                                <TableCell>{Number(item.relation_count ?? 0)}</TableCell>
                              </TableRow>
                            )
                          }) : (
                            <TableRow>
                              <TableCell colSpan={4} className="text-center text-muted-foreground">
                                当前没有可删除的来源
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <RotateCcw className="h-4 w-4" />
                      删除操作恢复
                    </CardTitle>
                    <CardDescription>按列表浏览最近的删除操作，先选中记录，再在下方确认影响范围并执行恢复。</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_180px_180px]">
                      <Input
                        value={operationSearch}
                        onChange={(event) => setOperationSearch(event.target.value)}
                        placeholder="搜索 operation / reason / requested_by / source"
                      />
                      <Select value={operationModeFilter} onValueChange={setOperationModeFilter}>
                        <SelectTrigger>
                          <SelectValue placeholder="按模式筛选" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">全部模式</SelectItem>
                          <SelectItem value="source">来源删除</SelectItem>
                          <SelectItem value="mixed">混合删除</SelectItem>
                          <SelectItem value="entity">实体删除</SelectItem>
                          <SelectItem value="relation">关系删除</SelectItem>
                          <SelectItem value="paragraph">段落删除</SelectItem>
                        </SelectContent>
                      </Select>
                      <Select value={operationStatusFilter} onValueChange={setOperationStatusFilter}>
                        <SelectTrigger>
                          <SelectValue placeholder="按状态筛选" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">全部状态</SelectItem>
                          <SelectItem value="executed">已执行</SelectItem>
                          <SelectItem value="restored">已恢复</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-muted-foreground">
                      <span>当前命中 {filteredDeleteOperations.length} 条记录，已加载最近 {deleteOperations.length} 条</span>
                      <span>第 {operationPage} / {deleteOperationPageCount} 页，每页显示 {DELETE_OPERATION_PAGE_SIZE} 条</span>
                    </div>

                    <ScrollArea className="h-[320px] rounded-lg border">
                      <div className="space-y-3 p-3">
                        {pagedDeleteOperations.length > 0 ? pagedDeleteOperations.map((operation) => {
                          const summary = (operation.summary ?? {}) as Record<string, unknown>
                          const counts = ((summary.counts as Record<string, number> | undefined) ?? {})
                          const isSelected = selectedDeleteOperation?.operation_id === operation.operation_id
                          return (
                            <button
                              key={operation.operation_id}
                              type="button"
                              onClick={() => setSelectedOperationId(operation.operation_id)}
                              className={cn(
                                'w-full rounded-xl border p-4 text-left transition-colors',
                                isSelected
                                  ? 'border-primary bg-primary/5 shadow-sm'
                                  : 'bg-muted/20 hover:border-primary/40 hover:bg-muted/40',
                              )}
                            >
                              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                <div className="min-w-0 space-y-2">
                                  <div className="flex flex-wrap items-center gap-2">
                                    <Badge variant={operation.status === 'restored' ? 'secondary' : 'default'}>
                                      {formatDeleteOperationStatus(String(operation.status ?? ''))}
                                    </Badge>
                                    <Badge variant="outline">
                                      {formatDeleteOperationMode(String(operation.mode ?? ''))}
                                    </Badge>
                                  </div>
                                  <div className="font-mono text-xs break-all">{operation.operation_id}</div>
                                  <div className="text-sm text-muted-foreground">
                                    {operation.reason || '未填写原因'}
                                  </div>
                                </div>
                                <div className="flex flex-wrap gap-2 text-xs text-muted-foreground lg:max-w-[280px] lg:justify-end">
                                  <span>实体 {Number(counts.entities ?? 0)}</span>
                                  <span>关系 {Number(counts.relations ?? 0)}</span>
                                  <span>段落 {Number(counts.paragraphs ?? 0)}</span>
                                  <span>来源 {Number(counts.sources ?? 0)}</span>
                                </div>
                              </div>
                              <div className="mt-3 text-xs text-muted-foreground">
                                {formatDeleteOperationTime(operation.created_at)}
                              </div>
                            </button>
                          )
                        }) : (
                          <div className="rounded-lg border border-dashed bg-muted/20 p-6 text-center text-sm text-muted-foreground">
                            当前筛选条件下没有删除操作。
                          </div>
                        )}
                      </div>
                    </ScrollArea>

                    <div className="flex items-center justify-between gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setOperationPage((current) => Math.max(1, current - 1))}
                        disabled={operationPage <= 1}
                      >
                        上一页
                      </Button>
                      <div className="text-xs text-muted-foreground">
                        支持按 operation、模式、状态、发起人和 source 检索
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setOperationPage((current) => Math.min(deleteOperationPageCount, current + 1))}
                        disabled={operationPage >= deleteOperationPageCount}
                      >
                        下一页
                      </Button>
                    </div>

                    <div className="rounded-xl border bg-muted/20 p-4">
                      {selectedDeleteOperation ? (
                        <div className="space-y-4">
                          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                            <div className="space-y-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <Badge variant={selectedDeleteOperation.status === 'restored' ? 'secondary' : 'default'}>
                                  {formatDeleteOperationStatus(String(selectedDeleteOperation.status ?? ''))}
                                </Badge>
                                <Badge variant="outline">
                                  {formatDeleteOperationMode(String(selectedDeleteOperation.mode ?? ''))}
                                </Badge>
                              </div>
                              <div className="font-mono text-xs break-all">{selectedDeleteOperation.operation_id}</div>
                              <div className="text-sm text-muted-foreground">
                                {selectedDeleteOperation.reason || '未填写删除原因'}
                              </div>
                            </div>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => void restoreDeleteOperation(selectedDeleteOperation.operation_id)}
                              disabled={selectedDeleteOperation.status === 'restored' || deleteRestoring}
                            >
                              <RotateCcw className="mr-2 h-4 w-4" />
                              {selectedDeleteOperation.status === 'restored' ? '已恢复' : '恢复这次删除'}
                            </Button>
                          </div>

                          <div className="grid gap-3 lg:grid-cols-4">
                            <div className="rounded-lg border bg-background/60 p-3">
                              <div className="text-xs text-muted-foreground">发起人</div>
                              <div className="mt-1 text-sm">{selectedDeleteOperation.requested_by || '-'}</div>
                            </div>
                            <div className="rounded-lg border bg-background/60 p-3">
                              <div className="text-xs text-muted-foreground">创建时间</div>
                              <div className="mt-1 text-sm">{formatDeleteOperationTime(selectedDeleteOperation.created_at)}</div>
                            </div>
                            <div className="rounded-lg border bg-background/60 p-3">
                              <div className="text-xs text-muted-foreground">恢复时间</div>
                              <div className="mt-1 text-sm">{formatDeleteOperationTime(selectedDeleteOperation.restored_at)}</div>
                            </div>
                            <div className="rounded-lg border bg-background/60 p-3">
                              <div className="text-xs text-muted-foreground">删除摘要</div>
                              <div className="mt-1 flex flex-wrap gap-2">
                                <Badge variant="outline">实体 {Number(selectedOperationCounts.entities ?? 0)}</Badge>
                                <Badge variant="outline">关系 {Number(selectedOperationCounts.relations ?? 0)}</Badge>
                                <Badge variant="outline">段落 {Number(selectedOperationCounts.paragraphs ?? 0)}</Badge>
                                <Badge variant="outline">来源 {Number(selectedOperationCounts.sources ?? 0)}</Badge>
                              </div>
                            </div>
                          </div>

                          {selectedOperationDetailLoading ? (
                            <div className="rounded-lg border bg-background/60 p-4 text-sm text-muted-foreground">
                              正在加载影响对象详情...
                            </div>
                          ) : null}

                          {selectedOperationDetailError ? (
                            <Alert variant="destructive">
                              <AlertDescription>{selectedOperationDetailError}</AlertDescription>
                            </Alert>
                          ) : null}

                          {selectedOperationSources.length > 0 ? (
                            <div className="space-y-2">
                              <div className="text-sm font-semibold">关联来源</div>
                              <div className="flex flex-wrap gap-2">
                                {selectedOperationSources.map((source) => (
                                  <Badge key={source} variant="secondary" className="max-w-full break-all">
                                    {source}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          ) : null}

                          <div className="grid gap-4 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
                            <div className="space-y-2">
                              <div className="text-sm font-semibold">选择器</div>
                              <pre className="max-h-56 overflow-auto rounded-lg border bg-background/70 p-3 text-xs break-words whitespace-pre-wrap">
                                {JSON.stringify(selectedDeleteOperation.selector ?? {}, null, 2)}
                              </pre>
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <div className="text-sm font-semibold">影响对象</div>
                                <div className="text-xs text-muted-foreground">
                                  命中 {filteredSelectedOperationItems.length} / {selectedOperationItems.length} 项
                                </div>
                              </div>
                              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                                <Input
                                  value={selectedOperationItemSearch}
                                  onChange={(event) => setSelectedOperationItemSearch(event.target.value)}
                                  placeholder="搜索类型 / hash / item_key / source"
                                  className="lg:max-w-sm"
                                />
                                <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground lg:min-w-[180px] lg:justify-end">
                                  <span>第 {selectedOperationItemPage} / {selectedOperationItemPageCount} 页</span>
                                  <span>每页 {DELETE_OPERATION_ITEM_PAGE_SIZE} 项</span>
                                </div>
                              </div>
                              <ScrollArea className="h-[280px] rounded-lg border bg-background/60">
                                <div className="space-y-2 p-3">
                                  {pagedSelectedOperationItems.length > 0 ? pagedSelectedOperationItems.map((item) => {
                                    const source = getDeleteOperationItemSource(item)
                                    const label = getDeleteOperationItemLabel(item)
                                    const preview = getDeleteOperationItemPreview(item)
                                    return (
                                      <div key={`${item.item_type}:${item.item_hash}:${item.item_key ?? ''}`} className="rounded-lg border bg-muted/20 p-3">
                                        <div className="flex flex-wrap items-center gap-2">
                                          <Badge variant="outline">{item.item_type}</Badge>
                                          {source ? <Badge variant="secondary">{source}</Badge> : null}
                                          {item.item_key && item.item_key !== item.item_hash ? (
                                            <span className="text-xs text-muted-foreground break-all">{item.item_key}</span>
                                          ) : null}
                                        </div>
                                        <div className="mt-2 text-sm font-medium break-words">
                                          {label}
                                        </div>
                                        {preview ? (
                                          <div className="mt-1 text-xs text-muted-foreground break-words">
                                            {preview}
                                          </div>
                                        ) : null}
                                        <div className="mt-2 font-mono text-[11px] break-all text-muted-foreground">
                                          {item.item_hash}
                                        </div>
                                      </div>
                                    )
                                  }) : (
                                    <div className="rounded-lg border border-dashed bg-muted/20 p-6 text-center text-sm text-muted-foreground">
                                      {selectedOperationItems.length > 0 ? '当前筛选条件下没有明细项。' : '当前操作没有记录明细项。'}
                                    </div>
                                  )}
                                </div>
                              </ScrollArea>
                              <div className="flex items-center justify-between gap-2">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setSelectedOperationItemPage((current) => Math.max(1, current - 1))}
                                  disabled={selectedOperationItemPage <= 1}
                                >
                                  上一页
                                </Button>
                                <div className="text-xs text-muted-foreground">
                                  支持按对象类型、hash、item_key、source 检索
                                </div>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setSelectedOperationItemPage((current) => Math.min(selectedOperationItemPageCount, current + 1))}
                                  disabled={selectedOperationItemPage >= selectedOperationItemPageCount}
                                >
                                  下一页
                                </Button>
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="flex min-h-[320px] items-center justify-center rounded-lg border border-dashed bg-background/40 p-6 text-center text-sm text-muted-foreground">
                          当前没有可查看的删除操作详情。
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      <MemoryDeleteDialog
        open={deleteDialogOpen}
        onOpenChange={closeDeleteDialog}
        title={deleteDialogTitle}
        description={deleteDialogDescription}
        preview={deletePreview}
        result={deleteResult}
        loadingPreview={deletePreviewLoading}
        executing={deleteExecuting}
        restoring={deleteRestoring}
        error={deletePreviewError}
        onExecute={() => void executePendingDelete()}
        onRestore={() => void (deleteResult?.operation_id ? restoreDeleteOperation(deleteResult.operation_id) : Promise.resolve())}
      />
    </div>
  )
}
