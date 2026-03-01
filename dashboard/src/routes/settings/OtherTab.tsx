import { AlertTriangle, Database, Download, HardDrive, RefreshCw, RotateCcw, Trash2, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'

import { cn } from '@/lib/utils'
import { fetchWithAuth } from '@/lib/fetch-with-auth'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { useToast } from '@/hooks/use-toast'
import { clearLocalCache, DEFAULT_SETTINGS, exportSettings, formatBytes, getSetting, getStorageUsage, importSettings, resetAllSettings, setSetting } from '@/lib/settings-manager'
import { logWebSocket } from '@/lib/log-websocket'
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog'

// 其他设置标签页
export function OtherTab() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [isResetting, setIsResetting] = useState(false)
  const [shouldThrowError, setShouldThrowError] = useState(false)
  
  // 性能与存储设置状态
  const [logCacheSize, setLogCacheSize] = useState(() => getSetting('logCacheSize'))
  const [wsReconnectInterval, setWsReconnectInterval] = useState(() => getSetting('wsReconnectInterval'))
  const [wsMaxReconnectAttempts, setWsMaxReconnectAttempts] = useState(() => getSetting('wsMaxReconnectAttempts'))
  const [dataSyncInterval, setDataSyncInterval] = useState(() => getSetting('dataSyncInterval'))
  const [storageUsage, setStorageUsage] = useState(() => getStorageUsage())
  
  // 导入/导出状态
  const [isExporting, setIsExporting] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 手动触发 React 错误
  if (shouldThrowError) {
    throw new Error('这是一个手动触发的测试错误，用于验证错误边界组件是否正常工作。')
  }

  // 刷新存储使用情况
  const refreshStorageUsage = () => {
    setStorageUsage(getStorageUsage())
  }

  // 处理日志缓存大小变更
  const handleLogCacheSizeChange = (value: number[]) => {
    const size = value[0]
    setLogCacheSize(size)
    setSetting('logCacheSize', size)
  }

  // 处理 WebSocket 重连间隔变更
  const handleWsReconnectIntervalChange = (value: number[]) => {
    const interval = value[0]
    setWsReconnectInterval(interval)
    setSetting('wsReconnectInterval', interval)
  }

  // 处理 WebSocket 最大重连次数变更
  const handleWsMaxReconnectAttemptsChange = (value: number[]) => {
    const attempts = value[0]
    setWsMaxReconnectAttempts(attempts)
    setSetting('wsMaxReconnectAttempts', attempts)
  }

  // 处理数据同步间隔变更
  const handleDataSyncIntervalChange = (value: number[]) => {
    const interval = value[0]
    setDataSyncInterval(interval)
    setSetting('dataSyncInterval', interval)
  }

  // 清除日志缓存
  const handleClearLogCache = () => {
    logWebSocket.clearLogs()
    toast({
      title: '日志已清除',
      description: '日志缓存已清空',
    })
  }

  // 清除本地缓存
  const handleClearLocalCache = () => {
    const result = clearLocalCache()
    refreshStorageUsage()
    toast({
      title: '缓存已清除',
      description: `已清除 ${result.clearedKeys.length} 项缓存数据`,
    })
  }

  // 导出设置
  const handleExportSettings = () => {
    setIsExporting(true)
    try {
      const settings = exportSettings()
      const dataStr = JSON.stringify(settings, null, 2)
      const blob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `maibot-webui-settings-${new Date().toISOString().slice(0, 10)}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast({
        title: '导出成功',
        description: '设置已导出为 JSON 文件',
      })
    } catch (error) {
      console.error('导出设置失败:', error)
      toast({
        title: '导出失败',
        description: '无法导出设置',
        variant: 'destructive',
      })
    } finally {
      setIsExporting(false)
    }
  }

  // 导入设置
  const handleImportSettings = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsImporting(true)
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string
        const settings = JSON.parse(content)
        const result = importSettings(settings)
        
        if (result.success) {
          // 刷新页面状态
          setLogCacheSize(getSetting('logCacheSize'))
          setWsReconnectInterval(getSetting('wsReconnectInterval'))
          setWsMaxReconnectAttempts(getSetting('wsMaxReconnectAttempts'))
          setDataSyncInterval(getSetting('dataSyncInterval'))
          refreshStorageUsage()
          
          toast({
            title: '导入成功',
            description: `成功导入 ${result.imported.length} 项设置${result.skipped.length > 0 ? `，跳过 ${result.skipped.length} 项` : ''}`,
          })
          
          // 提示用户刷新页面以应用所有更改
          if (result.imported.includes('theme') || result.imported.includes('accentColor')) {
            toast({
              title: '提示',
              description: '部分设置需要刷新页面才能完全生效',
            })
          }
        } else {
          toast({
            title: '导入失败',
            description: '没有有效的设置项可导入',
            variant: 'destructive',
          })
        }
      } catch (error) {
        console.error('导入设置失败:', error)
        toast({
          title: '导入失败',
          description: '文件格式无效',
          variant: 'destructive',
        })
      } finally {
        setIsImporting(false)
        // 清空 input，允许重复选择同一文件
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
      }
    }
    reader.readAsText(file)
  }

  // 重置所有设置
  const handleResetAllSettings = () => {
    resetAllSettings()
    // 刷新页面状态
    setLogCacheSize(DEFAULT_SETTINGS.logCacheSize)
    setWsReconnectInterval(DEFAULT_SETTINGS.wsReconnectInterval)
    setWsMaxReconnectAttempts(DEFAULT_SETTINGS.wsMaxReconnectAttempts)
    setDataSyncInterval(DEFAULT_SETTINGS.dataSyncInterval)
    refreshStorageUsage()
    toast({
      title: '已重置',
      description: '所有设置已恢复为默认值，刷新页面以应用更改',
    })
  }

  const handleResetSetup = async () => {
    setIsResetting(true)

    try {
      // 调用后端API重置首次配置状态
      const response = await fetchWithAuth('/api/webui/setup/reset', {
        method: 'POST',
      })

      const data = await response.json()

      if (response.ok && data.success) {
        toast({
          title: '重置成功',
          description: '即将进入初次配置向导',
        })

        // 延迟跳转到配置向导
        setTimeout(() => {
          navigate({ to: '/setup' })
        }, 1000)
      } else {
        toast({
          title: '重置失败',
          description: data.message || '无法重置配置状态',
          variant: 'destructive',
        })
      }
    } catch (error) {
      console.error('重置配置状态错误:', error)
      toast({
        title: '重置失败',
        description: '连接服务器失败',
        variant: 'destructive',
      })
    } finally {
      setIsResetting(false)
    }
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* 性能与存储 */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 flex items-center gap-2">
          <Database className="h-5 w-5" />
          性能与存储
        </h3>
        <div className="space-y-4 sm:space-y-5">
          {/* 存储使用情况 */}
          <div className="rounded-lg bg-muted/50 p-3 sm:p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium flex items-center gap-2">
                <HardDrive className="h-4 w-4" />
                本地存储使用
              </span>
              <Button variant="ghost" size="sm" onClick={refreshStorageUsage} className="h-7 px-2">
                <RefreshCw className="h-3 w-3" />
              </Button>
            </div>
            <div className="text-2xl font-bold text-primary">{formatBytes(storageUsage.used)}</div>
            <p className="text-xs text-muted-foreground mt-1">{storageUsage.items} 个存储项</p>
          </div>

          {/* 日志缓存大小 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">日志缓存大小</Label>
              <span className="text-sm text-muted-foreground">{logCacheSize} 条</span>
            </div>
            <Slider
              value={[logCacheSize]}
              onValueChange={handleLogCacheSizeChange}
              min={100}
              max={5000}
              step={100}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              控制日志查看器最多缓存的日志条数，较大的值会占用更多内存
            </p>
          </div>

          {/* 数据刷新间隔 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">首页数据刷新间隔</Label>
              <span className="text-sm text-muted-foreground">{dataSyncInterval} 秒</span>
            </div>
            <Slider
              value={[dataSyncInterval]}
              onValueChange={handleDataSyncIntervalChange}
              min={10}
              max={120}
              step={5}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              控制首页统计数据的自动刷新间隔
            </p>
          </div>

          {/* WebSocket 重连间隔 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">WebSocket 重连间隔</Label>
              <span className="text-sm text-muted-foreground">{wsReconnectInterval / 1000} 秒</span>
            </div>
            <Slider
              value={[wsReconnectInterval]}
              onValueChange={handleWsReconnectIntervalChange}
              min={1000}
              max={10000}
              step={500}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              日志 WebSocket 连接断开后的重连基础间隔
            </p>
          </div>

          {/* WebSocket 最大重连次数 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">WebSocket 最大重连次数</Label>
              <span className="text-sm text-muted-foreground">{wsMaxReconnectAttempts} 次</span>
            </div>
            <Slider
              value={[wsMaxReconnectAttempts]}
              onValueChange={handleWsMaxReconnectAttemptsChange}
              min={3}
              max={30}
              step={1}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              连接失败后的最大重连尝试次数
            </p>
          </div>

          {/* 清理按钮 */}
          <div className="flex flex-wrap gap-2 pt-2">
            <Button variant="outline" size="sm" onClick={handleClearLogCache} className="gap-2">
              <Trash2 className="h-4 w-4" />
              清除日志缓存
            </Button>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="gap-2">
                  <Trash2 className="h-4 w-4" />
                  清除本地缓存
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>确认清除本地缓存</AlertDialogTitle>
                  <AlertDialogDescription>
                    这将清除所有本地缓存的设置和数据（不包括登录凭证）。
                    您可能需要重新配置部分偏好设置。确定要继续吗？
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>取消</AlertDialogCancel>
                  <AlertDialogAction onClick={handleClearLocalCache}>
                    确认清除
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>
      </div>

      {/* 导入/导出设置 */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 flex items-center gap-2">
          <Download className="h-5 w-5" />
          导入/导出设置
        </h3>
        <div className="space-y-4">
          <p className="text-xs sm:text-sm text-muted-foreground">
            导出当前的界面设置以便备份，或从之前导出的文件中恢复设置。
          </p>
          
          <div className="flex flex-wrap gap-2">
            <Button 
              variant="outline" 
              onClick={handleExportSettings} 
              disabled={isExporting}
              className="gap-2"
            >
              <Download className="h-4 w-4" />
              {isExporting ? '导出中...' : '导出设置'}
            </Button>
            
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImportSettings}
              className="hidden"
            />
            <Button 
              variant="outline" 
              onClick={() => fileInputRef.current?.click()}
              disabled={isImporting}
              className="gap-2"
            >
              <Upload className="h-4 w-4" />
              {isImporting ? '导入中...' : '导入设置'}
            </Button>
          </div>

          {/* 重置所有设置 */}
          <div className="pt-2 border-t">
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="gap-2 text-destructive hover:text-destructive">
                  <RotateCcw className="h-4 w-4" />
                  重置所有设置为默认值
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>确认重置所有设置</AlertDialogTitle>
                  <AlertDialogDescription>
                    这将把所有界面设置恢复为默认值，包括主题、颜色、动画等偏好设置。
                    此操作不会影响您的登录状态。确定要继续吗？
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>取消</AlertDialogCancel>
                  <AlertDialogAction onClick={handleResetAllSettings}>
                    确认重置
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>
      </div>

      {/* 配置向导 */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">配置向导</h3>
        <div className="space-y-3 sm:space-y-4">
          <div className="space-y-2">
            <p className="text-xs sm:text-sm text-muted-foreground">
              重新进行初次配置向导，可以帮助您重新设置系统的基础配置。
            </p>
          </div>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" disabled={isResetting} className="gap-2">
                <RotateCcw className={cn('h-4 w-4', isResetting && 'animate-spin')} />
                重新进行初次配置
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>确认重新配置</AlertDialogTitle>
                <AlertDialogDescription>
                  这将带您重新进入初次配置向导。您可以重新设置系统的基础配置项。确定要继续吗？
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>取消</AlertDialogCancel>
                <AlertDialogAction onClick={handleResetSetup}>
                  确认重置
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      {/* 开发者工具 */}
      <div className="rounded-lg border border-dashed border-yellow-500/50 bg-yellow-500/5 p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-yellow-500" />
          开发者工具
        </h3>
        <div className="space-y-3 sm:space-y-4">
          <div className="space-y-2">
            <p className="text-xs sm:text-sm text-muted-foreground">
              以下功能仅供开发调试使用，可能会导致页面崩溃或异常。
            </p>
          </div>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" className="gap-2">
                <AlertTriangle className="h-4 w-4" />
                触发测试错误
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>确认触发错误</AlertDialogTitle>
                <AlertDialogDescription>
                  这将手动触发一个 React 错误，用于测试错误边界组件的显示效果。
                  页面将显示错误界面，您可以通过刷新页面或点击返回首页来恢复。
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>取消</AlertDialogCancel>
                <AlertDialogAction 
                  onClick={() => setShouldThrowError(true)}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  确认触发
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>
    </div>
  )
}
