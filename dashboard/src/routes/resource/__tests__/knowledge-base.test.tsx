import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { KnowledgeBasePage } from '../knowledge-base'
import * as memoryApi from '@/lib/memory-api'

const navigateMock = vi.fn()
const toastMock = vi.fn()

vi.mock('@tanstack/react-router', () => ({
  useNavigate: () => navigateMock,
}))

vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({ toast: toastMock }),
}))

vi.mock('@/components', () => ({
  CodeEditor: ({ value }: { value: string }) => <pre data-testid="code-editor">{value}</pre>,
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
}))

vi.mock('@/components/memory/MemoryConfigEditor', () => ({
  MemoryConfigEditor: () => <div data-testid="memory-config-editor">memory-config-editor</div>,
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

vi.mock('@/lib/memory-api', () => ({
  getMemoryConfigSchema: vi.fn(),
  getMemoryConfig: vi.fn(),
  getMemoryConfigRaw: vi.fn(),
  getMemoryDeleteOperation: vi.fn(),
  getMemoryRuntimeConfig: vi.fn(),
  getMemoryImportGuide: vi.fn(),
  getMemoryImportTasks: vi.fn(),
  getMemoryTuningProfile: vi.fn(),
  getMemoryTuningTasks: vi.fn(),
  getMemorySources: vi.fn(),
  getMemoryDeleteOperations: vi.fn(),
  refreshMemoryRuntimeSelfCheck: vi.fn(),
  updateMemoryConfig: vi.fn(),
  updateMemoryConfigRaw: vi.fn(),
  createMemoryPasteImport: vi.fn(),
  createMemoryTuningTask: vi.fn(),
  applyBestMemoryTuningProfile: vi.fn(),
  previewMemoryDelete: vi.fn(),
  executeMemoryDelete: vi.fn(),
  restoreMemoryDelete: vi.fn(),
}))

describe('KnowledgeBasePage', () => {
  beforeEach(() => {
    navigateMock.mockReset()
    toastMock.mockReset()

    vi.mocked(memoryApi.getMemoryConfigSchema).mockResolvedValue({
      success: true,
      path: 'config/a_memorix.toml',
      schema: {
        plugin_id: 'a_memorix',
        plugin_info: {
          name: 'A_Memorix',
          version: '2.0.0',
          description: '长期记忆子系统',
          author: 'A_Dawn',
        },
        _note: 'raw-only 字段仍可通过 TOML 编辑',
        layout: {
          type: 'tabs',
          tabs: [{ id: 'basic', title: '基础', sections: ['plugin'], order: 1 }],
        },
        sections: {
          plugin: {
            name: 'plugin',
            title: '子系统状态',
            collapsed: false,
            order: 1,
            fields: {},
          },
        },
      },
    })
    vi.mocked(memoryApi.getMemoryConfig).mockResolvedValue({
      success: true,
      path: 'config/a_memorix.toml',
      config: { plugin: { enabled: true } },
    })
    vi.mocked(memoryApi.getMemoryConfigRaw).mockResolvedValue({
      success: true,
      path: 'config/a_memorix.toml',
      config: '[plugin]\nenabled = true\n',
    })
    vi.mocked(memoryApi.getMemoryRuntimeConfig).mockResolvedValue({
      success: true,
      config: { plugin: { enabled: true } },
      data_dir: 'data/plugins/a-dawn.a-memorix',
      embedding_dimension: 1024,
      auto_save: true,
      relation_vectors_enabled: false,
      runtime_ready: true,
      embedding_degraded: false,
      embedding_degraded_reason: '',
      embedding_degraded_since: null,
      embedding_last_check: null,
      paragraph_vector_backfill_pending: 2,
      paragraph_vector_backfill_running: 0,
      paragraph_vector_backfill_failed: 1,
      paragraph_vector_backfill_done: 3,
    })
    vi.mocked(memoryApi.getMemoryImportGuide).mockResolvedValue({
      success: true,
      content: '# 导入指南\n导入说明',
    })
    vi.mocked(memoryApi.getMemoryImportTasks).mockResolvedValue({
      success: true,
      items: [{ task_id: 'import-1', status: 'done', mode: 'text' }],
    })
    vi.mocked(memoryApi.getMemoryTuningProfile).mockResolvedValue({
      success: true,
      profile: { retrieval: { top_k: 10 } },
      toml: '[retrieval]\ntop_k = 10\n',
    })
    vi.mocked(memoryApi.getMemoryTuningTasks).mockResolvedValue({
      success: true,
      items: [{ task_id: 'tune-1', status: 'done' }],
    })
    vi.mocked(memoryApi.getMemorySources).mockResolvedValue({
      success: true,
      items: [
        { source: 'demo-1', paragraph_count: 2, relation_count: 1 },
        { source: 'demo-2', paragraph_count: 1, relation_count: 0 },
      ],
      count: 2,
    })
    vi.mocked(memoryApi.getMemoryDeleteOperations).mockResolvedValue({
      success: true,
      items: [
        {
          operation_id: 'del-1',
          mode: 'source',
          status: 'executed',
          summary: { counts: { paragraphs: 2, relations: 1, sources: 1 } },
        },
      ],
      count: 1,
    })
    vi.mocked(memoryApi.getMemoryDeleteOperation).mockResolvedValue({
      success: true,
      operation: {
        operation_id: 'del-1',
        mode: 'source',
        status: 'executed',
        selector: { sources: ['demo-1'] },
        summary: { counts: { paragraphs: 2, relations: 1, sources: 1 }, sources: ['demo-1'] },
        items: [
          {
            item_type: 'paragraph',
            item_hash: 'p-1',
            item_key: 'paragraph:p-1',
            payload: { paragraph: { source: 'demo-1', content: '这是用于测试删除详情展示的段落内容。' } },
          },
        ],
      },
    })
    vi.mocked(memoryApi.refreshMemoryRuntimeSelfCheck).mockResolvedValue({
      success: true,
      report: { ok: true },
    })
    vi.mocked(memoryApi.updateMemoryConfig).mockResolvedValue({
      success: true,
      config_path: 'config/a_memorix.toml',
    } as never)
    vi.mocked(memoryApi.updateMemoryConfigRaw).mockResolvedValue({
      success: true,
      config_path: 'config/a_memorix.toml',
    } as never)
    vi.mocked(memoryApi.createMemoryPasteImport).mockResolvedValue({ success: true } as never)
    vi.mocked(memoryApi.createMemoryTuningTask).mockResolvedValue({ success: true } as never)
    vi.mocked(memoryApi.applyBestMemoryTuningProfile).mockResolvedValue({ success: true } as never)
    vi.mocked(memoryApi.previewMemoryDelete).mockResolvedValue({
      success: true,
      mode: 'source',
      selector: { sources: ['demo-1'] },
      counts: { sources: 1, paragraphs: 2, relations: 1 },
      sources: ['demo-1'],
      items: [{ item_type: 'paragraph', item_hash: 'p-1', label: 'demo-1' }],
      item_count: 1,
      dry_run: true,
    } as never)
    vi.mocked(memoryApi.executeMemoryDelete).mockResolvedValue({
      success: true,
      mode: 'source',
      operation_id: 'del-2',
      counts: { sources: 1, paragraphs: 2, relations: 1 },
      sources: ['demo-1'],
      deleted_count: 4,
      deleted_entity_count: 0,
      deleted_relation_count: 1,
      deleted_paragraph_count: 2,
      deleted_source_count: 1,
    } as never)
    vi.mocked(memoryApi.restoreMemoryDelete).mockResolvedValue({ success: true } as never)
  })

  it('renders long-term memory console and key tabs', async () => {
    const user = userEvent.setup()

    render(<KnowledgeBasePage />)

    expect(await screen.findByText('长期记忆控制台')).toBeInTheDocument()
    expect(screen.getByText(/config\/a_memorix\.toml/)).toBeInTheDocument()
    expect(screen.getByText('运行就绪')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: '配置' }))
    expect(await screen.findByTestId('memory-config-editor')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: '导入' }))
    expect(await screen.findByText(/导入说明/)).toBeInTheDocument()
    expect(screen.getByText('import-1')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: '调优' }))
    expect(await screen.findByText('tune-1')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '应用最佳' })).toBeInTheDocument()
  })

  it('shows delete tab and opens source delete preview', async () => {
    const user = userEvent.setup()

    render(<KnowledgeBasePage />)

    expect(await screen.findByText('长期记忆控制台')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '删除' }))

    expect(await screen.findByText('来源批量删除')).toBeInTheDocument()
    expect(screen.getAllByText('demo-1').length).toBeGreaterThan(0)
    expect(screen.getAllByText('del-1').length).toBeGreaterThan(0)
    expect(screen.getByText('恢复这次删除')).toBeInTheDocument()

    await user.click(screen.getAllByRole('checkbox')[0])
    await user.click(screen.getByRole('button', { name: '预览删除' }))

    expect(await screen.findByTestId('memory-delete-dialog')).toHaveTextContent('delete:source:1')
  })

  it('loads selected delete operation detail items from detail endpoint', async () => {
    const user = userEvent.setup()

    render(<KnowledgeBasePage />)

    expect(await screen.findByText('长期记忆控制台')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '删除' }))

    expect(await screen.findByText('删除操作恢复')).toBeInTheDocument()
    expect(await screen.findByText('paragraph')).toBeInTheDocument()
    expect(screen.getByText('p-1')).toBeInTheDocument()
    expect(screen.getByText('这是用于测试删除详情展示的段落内容。')).toBeInTheDocument()
  })
})
