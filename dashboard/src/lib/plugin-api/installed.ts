import type { ApiResponse } from '@/types/api'

import { fetchWithAuth, getAuthHeaders } from '@/lib/fetch-with-auth'
import { parseResponse } from '@/lib/api-helpers'

import type { InstalledPlugin } from './types'

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
  
  // 兼容两种格式：新格式有 manifest,旧格式直接有 version
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return plugin.manifest?.version || (plugin as any).version
}
