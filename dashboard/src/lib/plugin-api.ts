import type { ApiResponse } from '@/types/api'
import type { PluginInfo } from '@/types/plugin'

import { fetchWithAuth, getAuthHeaders } from '@/lib/fetch-with-auth'
import { parseResponse } from './api-helpers'
import { createReconnectingWebSocket } from './ws-utils'

/**
 * Git 安装状态
 */
export interface GitStatus {
  installed: boolean
  version?: string
  path?: string
  error?: string
}

/**
 * 麦麦版本信息
 */
export interface MaimaiVersion {
  version: string
  version_major: number
  version_minor: number
  version_patch: number
}

/**
 * 已安装插件信息
 */
export interface InstalledPlugin {
  id: string
  manifest: {
    manifest_version: number
    name: string
    version: string
    description: string
    author: {
      name: string
      url?: string
    }
    license: string
    host_application: {
      min_version: string
      max_version?: string
    }
    homepage_url?: string
    repository_url?: string
    keywords?: string[]
    categories?: string[]
    [key: string]: unknown  // 允许其他字段
  }
  path: string
}

/**
 * 插件加载进度
 */
export interface PluginLoadProgress {
  operation: 'idle' | 'fetch' | 'install' | 'uninstall' | 'update'
  stage: 'idle' | 'loading' | 'success' | 'error'
  progress: number  // 0-100
  message: string
  error?: string
  plugin_id?: string
  total_plugins: number
  loaded_plugins: number
}

/**
 * 插件仓库配置
 */
const PLUGIN_REPO_OWNER = 'Mai-with-u'
const PLUGIN_REPO_NAME = 'plugin-repo'
const PLUGIN_REPO_BRANCH = 'main'
const PLUGIN_DETAILS_FILE = 'plugin_details.json'

/**
 * 插件列表 API 响应类型（只包含我们需要的字段）
 */
interface PluginApiResponse {
  id: string
  manifest: {
    manifest_version: number
    name: string
    version: string
    description: string
    author: {
      name: string
      url?: string
    }
    license: string
    host_application: {
      min_version: string
      max_version?: string
    }
    homepage_url?: string
    repository_url?: string
    keywords: string[]
    categories?: string[]
    default_locale: string
    locales_path?: string
  }
  // 可能还有其他字段，但我们不关心
  [key: string]: unknown
}

/**
 * 从远程获取插件列表（通过后端代理避免 CORS）
 */
export async function fetchPluginList(): Promise<ApiResponse<PluginInfo[]>> {
  const response = await fetchWithAuth('/api/webui/plugins/fetch-raw', {
    method: 'POST',
    body: JSON.stringify({
      owner: PLUGIN_REPO_OWNER,
      repo: PLUGIN_REPO_NAME,
      branch: PLUGIN_REPO_BRANCH,
      file_path: PLUGIN_DETAILS_FILE
    })
  })
  
  const apiResult = await parseResponse<{ success: boolean; data: string; error?: string }>(response)
  
  if (!apiResult.success) {
    return apiResult
  }
  
  const result = apiResult.data
  if (!result.success || !result.data) {
    return {
      success: false,
      error: result.error || '获取插件列表失败'
    }
  }
  
  const data: PluginApiResponse[] = JSON.parse(result.data)
  
  const pluginList = data
    .filter(item => {
      if (!item?.id || !item?.manifest) {
        console.warn('跳过无效插件数据:', item)
        return false
      }
      if (!item.manifest.name || !item.manifest.version) {
        console.warn('跳过缺少必需字段的插件:', item.id)
        return false
      }
      return true
    })
    .map((item) => ({
      id: item.id,
      manifest: {
        manifest_version: item.manifest.manifest_version || 1,
        name: item.manifest.name,
        version: item.manifest.version,
        description: item.manifest.description || '',
        author: item.manifest.author || { name: 'Unknown' },
        license: item.manifest.license || 'Unknown',
        host_application: item.manifest.host_application || { min_version: '0.0.0' },
        homepage_url: item.manifest.homepage_url,
        repository_url: item.manifest.repository_url,
        keywords: item.manifest.keywords || [],
        categories: item.manifest.categories || [],
        default_locale: item.manifest.default_locale || 'zh-CN',
        locales_path: item.manifest.locales_path,
      },
      downloads: 0,
      rating: 0,
      review_count: 0,
      installed: false,
      published_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }))
  
  return {
    success: true,
    data: pluginList
  }
}

/**
 * 检查本机 Git 安装状态
 */
export async function checkGitStatus(): Promise<ApiResponse<GitStatus>> {
  const response = await fetchWithAuth('/api/webui/plugins/git-status')
  
  const apiResult = await parseResponse<GitStatus>(response)
  
  if (!apiResult.success) {
    return {
      success: true,
      data: {
        installed: false,
        error: '无法检测 Git 安装状态'
      }
    }
  }
  
  return apiResult
}

/**
 * 获取麦麦版本信息
 */
export async function getMaimaiVersion(): Promise<ApiResponse<MaimaiVersion>> {
  const response = await fetchWithAuth('/api/webui/plugins/version')
  
  const apiResult = await parseResponse<MaimaiVersion>(response)
  
  if (!apiResult.success) {
    return {
      success: true,
      data: {
        version: '0.0.0',
        version_major: 0,
        version_minor: 0,
        version_patch: 0
      }
    }
  }
  
  return apiResult
}

/**
 * 比较版本号
 * 
 * @param pluginMinVersion 插件要求的最小版本
 * @param pluginMaxVersion 插件要求的最大版本（可选）
 * @param maimaiVersion 麦麦当前版本
 * @returns true 表示兼容，false 表示不兼容
 */
export function isPluginCompatible(
  pluginMinVersion: string,
  pluginMaxVersion: string | undefined,
  maimaiVersion: MaimaiVersion
): boolean {
  // 解析插件最小版本
  const minParts = pluginMinVersion.split('.').map(p => parseInt(p) || 0)
  const minMajor = minParts[0] || 0
  const minMinor = minParts[1] || 0
  const minPatch = minParts[2] || 0
  
  // 检查最小版本
  if (maimaiVersion.version_major < minMajor) return false
  if (maimaiVersion.version_major === minMajor && maimaiVersion.version_minor < minMinor) return false
  if (maimaiVersion.version_major === minMajor && 
      maimaiVersion.version_minor === minMinor && 
      maimaiVersion.version_patch < minPatch) return false
  
  // 检查最大版本（如果有）
  if (pluginMaxVersion) {
    const maxParts = pluginMaxVersion.split('.').map(p => parseInt(p) || 0)
    const maxMajor = maxParts[0] || 0
    const maxMinor = maxParts[1] || 0
    const maxPatch = maxParts[2] || 0
    
    if (maimaiVersion.version_major > maxMajor) return false
    if (maimaiVersion.version_major === maxMajor && maimaiVersion.version_minor > maxMinor) return false
    if (maimaiVersion.version_major === maxMajor && 
        maimaiVersion.version_minor === maxMinor && 
        maimaiVersion.version_patch > maxPatch) return false
  }
  
  return true
}

/**
 * 连接插件加载进度 WebSocket
 * 
 * 使用临时 token 进行认证，异步获取 token 后连接
 */
export async function connectPluginProgressWebSocket(
  onProgress: (progress: PluginLoadProgress) => void,
  onError?: (error: Event) => void
): Promise<WebSocket | null> {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host
  const wsUrl = `${protocol}//${host}/api/webui/ws/plugin-progress`
  
  // 使用 ws-utils 创建 WebSocket
  const wsControl = createReconnectingWebSocket(wsUrl, {
    onMessage: (data: string) => {
      try {
        const progressData = JSON.parse(data) as PluginLoadProgress
        onProgress(progressData)
      } catch (error) {
        console.error('Failed to parse progress data:', error)
      }
    },
    onOpen: () => {
      console.log('Plugin progress WebSocket connected')
    },
    onClose: () => {
      console.log('Plugin progress WebSocket disconnected')
    },
    onError: (error) => {
      console.error('Plugin progress WebSocket error:', error)
      onError?.(error)
    },
    heartbeatInterval: 30000,
    maxRetries: 10,
    backoffBase: 1000,
    maxBackoff: 30000,
  })
  
  // 启动连接
  await wsControl.connect()
  
  // 返回 WebSocket 实例（用于外部检查连接状态）
  return wsControl.getWebSocket()
}

/**
 * 获取已安装插件列表
 */
export async function getInstalledPlugins(): Promise<ApiResponse<InstalledPlugin[]>> {
  const response = await fetchWithAuth('/api/webui/plugins/installed', {
    headers: getAuthHeaders()
  })
  
  const apiResult = await parseResponse<{ success: boolean; plugins?: InstalledPlugin[]; message?: string }>(response)
  
  if (!apiResult.success) {
    return {
      success: true,
      data: []
    }
  }
  
  const result = apiResult.data
  if (!result.success) {
    return {
      success: true,
      data: []
    }
  }
  
  return {
    success: true,
    data: result.plugins || []
  }
}

/**
 * 检查插件是否已安装
 */
export function checkPluginInstalled(pluginId: string, installedPlugins: InstalledPlugin[]): boolean {
  return installedPlugins.some(p => p.id === pluginId)
}

/**
 * 获取已安装插件的版本
 */
export function getInstalledPluginVersion(pluginId: string, installedPlugins: InstalledPlugin[]): string | undefined {
  const plugin = installedPlugins.find(p => p.id === pluginId)
  if (!plugin) return undefined
  
  // 兼容两种格式：新格式有 manifest，旧格式直接有 version
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return plugin.manifest?.version || (plugin as any).version
}

/**
 * 安装插件
 */
export async function installPlugin(pluginId: string, repositoryUrl: string, branch: string = 'main'): Promise<ApiResponse<{ success: boolean; message: string }>> {
  const response = await fetchWithAuth('/api/webui/plugins/install', {
    method: 'POST',
    body: JSON.stringify({
      plugin_id: pluginId,
      repository_url: repositoryUrl,
      branch: branch
    })
  })
  
  return await parseResponse<{ success: boolean; message: string }>(response)
}

/**
 * 卸载插件
 */
export async function uninstallPlugin(pluginId: string): Promise<ApiResponse<{ success: boolean; message: string }>> {
  const response = await fetchWithAuth('/api/webui/plugins/uninstall', {
    method: 'POST',
    body: JSON.stringify({
      plugin_id: pluginId
    })
  })
  
  return await parseResponse<{ success: boolean; message: string }>(response)
}

/**
 * 更新插件
 */
export async function updatePlugin(pluginId: string, repositoryUrl: string, branch: string = 'main'): Promise<ApiResponse<{ success: boolean; message: string; old_version: string; new_version: string }>> {
  const response = await fetchWithAuth('/api/webui/plugins/update', {
    method: 'POST',
    body: JSON.stringify({
      plugin_id: pluginId,
      repository_url: repositoryUrl,
      branch: branch
    })
  })
  
  return await parseResponse<{ success: boolean; message: string; old_version: string; new_version: string }>(response)
}


// ============ 插件配置管理 ============

/**
 * 列表项字段定义（用于 object 类型的数组项）
 */
export interface ItemFieldDefinition {
  type: string
  label?: string
  placeholder?: string
  default?: unknown
}

/**
 * 配置字段定义
 */
export interface ConfigFieldSchema {
  name: string
  type: string
  default: unknown
  description: string
  example?: string
  required: boolean
  choices?: unknown[]
  min?: number
  max?: number
  step?: number
  pattern?: string
  max_length?: number
  label: string
  placeholder?: string
  hint?: string
  icon?: string
  hidden: boolean
  disabled: boolean
  order: number
  input_type?: string
  ui_type: string
  rows?: number
  group?: string
  depends_on?: string
  depends_value?: unknown
  // 列表类型专用
  item_type?: string  // "string" | "number" | "object"
  item_fields?: Record<string, ItemFieldDefinition>
  min_items?: number
  max_items?: number
}

/**
 * 配置节定义
 */
export interface ConfigSectionSchema {
  name: string
  title: string
  description?: string
  icon?: string
  collapsed: boolean
  order: number
  fields: Record<string, ConfigFieldSchema>
}

/**
 * 配置标签页定义
 */
export interface ConfigTabSchema {
  id: string
  title: string
  sections: string[]
  icon?: string
  order: number
  badge?: string
}

/**
 * 配置布局定义
 */
export interface ConfigLayoutSchema {
  type: 'auto' | 'tabs' | 'pages'
  tabs: ConfigTabSchema[]
}

/**
 * 插件配置 Schema
 */
export interface PluginConfigSchema {
  plugin_id: string
  plugin_info: {
    name: string
    version: string
    description: string
    author: string
  }
  sections: Record<string, ConfigSectionSchema>
  layout: ConfigLayoutSchema
  _note?: string
}

/**
 * 获取插件配置 Schema
 */
export async function getPluginConfigSchema(pluginId: string): Promise<ApiResponse<PluginConfigSchema>> {
  const response = await fetchWithAuth(`/api/webui/plugins/config/${pluginId}/schema`, {
    headers: getAuthHeaders()
  })
  
  const apiResult = await parseResponse<{ success: boolean; schema?: PluginConfigSchema; message?: string }>(response)
  
  if (!apiResult.success) {
    return apiResult
  }
  
  const result = apiResult.data
  if (!result.success || !result.schema) {
    return {
      success: false,
      error: result.message || '获取配置 Schema 失败'
    }
  }
  
  return {
    success: true,
    data: result.schema
  }
}

/**
 * 获取插件当前配置值
 */
export async function getPluginConfig(pluginId: string): Promise<ApiResponse<Record<string, unknown>>> {
  const response = await fetchWithAuth(`/api/webui/plugins/config/${pluginId}`, {
    headers: getAuthHeaders()
  })
  
  const apiResult = await parseResponse<{ success: boolean; config?: Record<string, unknown>; message?: string }>(response)
  
  if (!apiResult.success) {
    return apiResult
  }
  
  const result = apiResult.data
  if (!result.success || !result.config) {
    return {
      success: false,
      error: result.message || '获取配置失败'
    }
  }
  
  return {
    success: true,
    data: result.config
  }
}

/**
 * 获取插件原始 TOML 配置
 */
export async function getPluginConfigRaw(pluginId: string): Promise<ApiResponse<string>> {
  const response = await fetchWithAuth(`/api/webui/plugins/config/${pluginId}/raw`, {
    headers: getAuthHeaders()
  })
  
  const apiResult = await parseResponse<{ success: boolean; config?: string; message?: string }>(response)
  
  if (!apiResult.success) {
    return apiResult
  }
  
  const result = apiResult.data
  if (!result.success || !result.config) {
    return {
      success: false,
      error: result.message || '获取配置失败'
    }
  }
  
  return {
    success: true,
    data: result.config
  }
}

/**
 * 更新插件配置
 */
export async function updatePluginConfig(
  pluginId: string,
  config: Record<string, unknown>
): Promise<ApiResponse<{ success: boolean; message: string; note?: string }>> {
  const response = await fetchWithAuth(`/api/webui/plugins/config/${pluginId}`, {
    method: 'PUT',
    headers: getAuthHeaders(),
    body: JSON.stringify({ config })
  })
  
  return await parseResponse<{ success: boolean; message: string; note?: string }>(response)
}

/**
 * 更新插件原始 TOML 配置
 */
export async function updatePluginConfigRaw(
  pluginId: string,
  configToml: string
): Promise<ApiResponse<{ success: boolean; message: string; note?: string }>> {
  const response = await fetchWithAuth(`/api/webui/plugins/config/${pluginId}/raw`, {
    method: 'PUT',
    headers: getAuthHeaders(),
    body: JSON.stringify({ config: configToml })
  })
  
  return await parseResponse<{ success: boolean; message: string; note?: string }>(response)
}

/**
 * 重置插件配置为默认值
 */
export async function resetPluginConfig(
  pluginId: string
): Promise<ApiResponse<{ success: boolean; message: string; backup?: string }>> {
  const response = await fetchWithAuth(`/api/webui/plugins/config/${pluginId}/reset`, {
    method: 'POST',
    headers: getAuthHeaders()
  })
  
  return await parseResponse<{ success: boolean; message: string; backup?: string }>(response)
}

/**
 * 切换插件启用状态
 */
export async function togglePlugin(
  pluginId: string
): Promise<ApiResponse<{ success: boolean; enabled: boolean; message: string; note?: string }>> {
  const response = await fetchWithAuth(`/api/webui/plugins/config/${pluginId}/toggle`, {
    method: 'POST',
    headers: getAuthHeaders()
  })
  
  return await parseResponse<{ success: boolean; enabled: boolean; message: string; note?: string }>(response)
}
