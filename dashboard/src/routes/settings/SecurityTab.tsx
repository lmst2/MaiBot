import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Copy,
  Eye,
  EyeOff,
  RefreshCw,
  XCircle,
} from 'lucide-react'
import { useState, useMemo } from 'react'

import { useNavigate } from '@tanstack/react-router'
import { cn } from '@/lib/utils'
import { useToast } from '@/hooks/use-toast'
import { validateToken } from '@/lib/token-validator'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'

export function SecurityTab() {
  const navigate = useNavigate()
  const [currentToken, setCurrentToken] = useState('')
  const [newToken, setNewToken] = useState('')
  const [showCurrentToken, setShowCurrentToken] = useState(false)
  const [showNewToken, setShowNewToken] = useState(false)
  const [isUpdating, setIsUpdating] = useState(false)
  const [isRegenerating, setIsRegenerating] = useState(false)
  const [copied, setCopied] = useState(false)
  const [showTokenDialog, setShowTokenDialog] = useState(false)
  const [generatedToken, setGeneratedToken] = useState('')
  const [tokenCopied, setTokenCopied] = useState(false)
  const { toast } = useToast()

  // 实时验证新 Token
  const tokenValidation = useMemo(() => validateToken(newToken), [newToken])

  // 复制 token 到剪贴板
  const copyToClipboard = async (text: string) => {
    if (!currentToken) {
      toast({
        title: '无法复制',
        description: 'Token 存储在安全 Cookie 中，请重新生成以获取新 Token',
        variant: 'destructive',
      })
      return
    }
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      toast({
        title: '复制成功',
        description: 'Token 已复制到剪贴板',
      })
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast({
        title: '复制失败',
        description: '请手动复制 Token',
        variant: 'destructive',
      })
    }
  }

  // 更新 token
  const handleUpdateToken = async () => {
    if (!newToken.trim()) {
      toast({
        title: '输入错误',
        description: '请输入新的 Token',
        variant: 'destructive',
      })
      return
    }

    // 验证 Token 格式
    if (!tokenValidation.isValid) {
      const failedRules = tokenValidation.rules
        .filter((rule) => !rule.passed)
        .map((rule) => rule.label)
        .join(', ')
      
      toast({
        title: '格式错误',
        description: `Token 不符合要求: ${failedRules}`,
        variant: 'destructive',
      })
      return
    }

    setIsUpdating(true)

    try {
      const response = await fetch('/api/webui/auth/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // 使用 Cookie 认证
        body: JSON.stringify({ new_token: newToken.trim() }),
      })

      const data = await response.json()

      if (response.ok && data.success) {
        // 清空输入框
        setNewToken('')
        
        // 更新当前显示的 Token
        setCurrentToken(newToken.trim())
        
        toast({
          title: '更新成功',
          description: 'Access Token 已更新，即将跳转到登录页',
        })

        // 延迟跳转到登录页
        setTimeout(() => {
          navigate({ to: '/auth' })
        }, 1500)
      } else {
        toast({
          title: '更新失败',
          description: data.message || '无法更新 Token',
          variant: 'destructive',
        })
      }
    } catch (err) {
      console.error('更新 Token 错误:', err)
      toast({
        title: '更新失败',
        description: '连接服务器失败',
        variant: 'destructive',
      })
    } finally {
      setIsUpdating(false)
    }
  }

  // 重新生成 token (实际执行函数)
  const executeRegenerateToken = async () => {
    setIsRegenerating(true)

    try {
      const response = await fetch('/api/webui/auth/regenerate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // 使用 Cookie 认证
      })

      const data = await response.json()

      if (response.ok && data.success) {
        // 更新当前显示的 Token
        setCurrentToken(data.token)
        
        // 显示弹窗展示新 Token
        setGeneratedToken(data.token)
        setShowTokenDialog(true)
        setTokenCopied(false)
        
        toast({
          title: '生成成功',
          description: '新的 Access Token 已生成，请及时保存',
        })
      } else {
        toast({
          title: '生成失败',
          description: data.message || '无法生成新 Token',
          variant: 'destructive',
        })
      }
    } catch (err) {
      console.error('生成 Token 错误:', err)
      toast({
        title: '生成失败',
        description: '连接服务器失败',
        variant: 'destructive',
      })
    } finally {
      setIsRegenerating(false)
    }
  }

  // 复制生成的 Token
  const copyGeneratedToken = async () => {
    try {
      await navigator.clipboard.writeText(generatedToken)
      setTokenCopied(true)
      toast({
        title: '复制成功',
        description: 'Token 已复制到剪贴板',
      })
    } catch {
      toast({
        title: '复制失败',
        description: '请手动复制 Token',
        variant: 'destructive',
      })
    }
  }

  // 关闭弹窗
  const handleCloseDialog = () => {
    setShowTokenDialog(false)
    // 延迟清空 token，避免用户看到内容消失
    setTimeout(() => {
      setGeneratedToken('')
      setTokenCopied(false)
    }, 300)
    
    // 跳转到登录页
    setTimeout(() => {
      navigate({ to: '/auth' })
    }, 500)
  }

  // 处理对话框状态变化（包括点击外部、ESC 等关闭方式）
  const handleDialogOpenChange = (open: boolean) => {
    if (!open) {
      handleCloseDialog()
    }
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Token 生成成功弹窗 */}
      <Dialog open={showTokenDialog} onOpenChange={handleDialogOpenChange}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
              新的 Access Token
            </DialogTitle>
            <DialogDescription>
              这是您的新 Token，请立即保存。关闭此窗口后将跳转到登录页面。
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {/* Token 显示区域 */}
            <div className="rounded-lg border-2 border-primary/20 bg-primary/5 p-4">
              <Label className="text-xs text-muted-foreground mb-2 block">
                您的新 Token (64位安全令牌)
              </Label>
              <div className="font-mono text-sm break-all select-all bg-background p-3 rounded border">
                {generatedToken}
              </div>
            </div>

            {/* 警告提示 */}
            <div className="rounded-lg border border-yellow-200 dark:border-yellow-900 bg-yellow-50 dark:bg-yellow-950/30 p-3">
              <div className="flex gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-500 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-yellow-800 dark:text-yellow-300 space-y-1">
                  <p className="font-semibold">重要提示</p>
                  <ul className="list-disc list-inside space-y-0.5 text-xs">
                    <li>此 Token 仅显示一次，关闭后无法再查看</li>
                    <li>请立即复制并保存到安全的位置</li>
                    <li>关闭窗口后将自动跳转到登录页面</li>
                    <li>请使用新 Token 重新登录系统</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <DialogFooter className="gap-2 sm:gap-0">
            <Button
              variant="outline"
              onClick={copyGeneratedToken}
              className="gap-2"
            >
              {tokenCopied ? (
                <>
                  <Check className="h-4 w-4 text-green-500" />
                  已复制
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4" />
                  复制 Token
                </>
              )}
            </Button>
            <Button onClick={handleCloseDialog}>
              我已保存，关闭
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 当前 Token */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">当前 Access Token</h3>
        <div className="space-y-3 sm:space-y-4">
          <div className="space-y-2">
            <Label htmlFor="current-token" className="text-sm">您的访问令牌</Label>
            <div className="flex flex-col sm:flex-row gap-2">
              <div className="relative flex-1">
                <Input
                  id="current-token"
                  type={showCurrentToken ? 'text' : 'password'}
                  value={currentToken || '••••••••••••••••••••••••••••••••'}
                  readOnly
                  className="pr-10 font-mono text-sm"
                  placeholder="Token 存储在安全 Cookie 中"
                />
                <button
                  onClick={() => {
                    if (currentToken) {
                      setShowCurrentToken(!showCurrentToken)
                    } else {
                      toast({
                        title: '无法查看',
                        description: 'Token 存储在安全 Cookie 中，如需新 Token 请点击"重新生成"',
                      })
                    }
                  }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 hover:bg-accent rounded"
                  title={showCurrentToken ? '隐藏' : '显示'}
                >
                  {showCurrentToken ? (
                    <EyeOff className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <Eye className="h-4 w-4 text-muted-foreground" />
                  )}
                </button>
              </div>
              <div className="flex gap-2 w-full sm:w-auto">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => copyToClipboard(currentToken)}
                  title="复制到剪贴板"
                  className="flex-shrink-0"
                  disabled={!currentToken}
                >
                  {copied ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button
                      variant="outline"
                      disabled={isRegenerating}
                      className="gap-2 flex-1 sm:flex-none"
                    >
                      <RefreshCw className={cn('h-4 w-4', isRegenerating && 'animate-spin')} />
                      <span className="hidden sm:inline">重新生成</span>
                      <span className="sm:hidden">生成</span>
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>确认重新生成 Token</AlertDialogTitle>
                      <AlertDialogDescription>
                        这将生成一个新的 64 位安全令牌，并使当前 Token 立即失效。
                        您需要使用新 Token 重新登录系统。此操作不可撤销，确定要继续吗？
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>取消</AlertDialogCancel>
                      <AlertDialogAction onClick={executeRegenerateToken}>
                        确认生成
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
            <p className="text-[10px] sm:text-xs text-muted-foreground">
              请妥善保管您的 Access Token，不要泄露给他人
            </p>
          </div>
        </div>
      </div>

      {/* 更新 Token */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">自定义 Access Token</h3>
        <div className="space-y-3 sm:space-y-4">
          <div className="space-y-2">
            <Label htmlFor="new-token" className="text-sm">新的访问令牌</Label>
            <div className="relative">
              <Input
                id="new-token"
                type={showNewToken ? 'text' : 'password'}
                value={newToken}
                onChange={(e) => setNewToken(e.target.value)}
                className="pr-10 font-mono text-sm"
                placeholder="输入自定义 Token"
              />
              <button
                onClick={() => setShowNewToken(!showNewToken)}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 hover:bg-accent rounded"
                title={showNewToken ? '隐藏' : '显示'}
              >
                {showNewToken ? (
                  <EyeOff className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <Eye className="h-4 w-4 text-muted-foreground" />
                )}
              </button>
            </div>
            
            {/* Token 验证规则显示 */}
            {newToken && (
              <div className="mt-3 space-y-2 p-3 rounded-lg bg-muted/50">
                <p className="text-sm font-medium text-foreground">Token 安全要求:</p>
                <div className="space-y-1.5">
                  {tokenValidation.rules.map((rule) => (
                    <div key={rule.id} className="flex items-center gap-2 text-sm">
                      {rule.passed ? (
                        <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
                      ) : (
                        <XCircle className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      )}
                      <span className={cn(
                        rule.passed ? 'text-green-600 dark:text-green-400' : 'text-muted-foreground'
                      )}>
                        {rule.label}
                      </span>
                    </div>
                  ))}
                </div>
                {tokenValidation.isValid && (
                  <div className="mt-2 pt-2 border-t border-border">
                    <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
                      <Check className="h-4 w-4" />
                      <span className="font-medium">Token 格式正确，可以使用</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          <Button 
            onClick={handleUpdateToken} 
            disabled={isUpdating || !tokenValidation.isValid || !newToken} 
            className="w-full sm:w-auto"
          >
            {isUpdating ? '更新中...' : '更新自定义 Token'}
          </Button>
        </div>
      </div>

      {/* 安全提示 */}
      <div className="rounded-lg border border-yellow-200 dark:border-yellow-900 bg-yellow-50 dark:bg-yellow-950/30 p-3 sm:p-4">
        <h4 className="text-sm sm:text-base font-semibold text-yellow-900 dark:text-yellow-200 mb-2">安全提示</h4>
        <ul className="text-xs sm:text-sm text-yellow-800 dark:text-yellow-300 space-y-1 list-disc list-inside">
          <li>重新生成 Token 会创建系统随机生成的 64 位安全令牌</li>
          <li>自定义 Token 必须满足所有安全要求才能使用</li>
          <li>更新 Token 后，旧的 Token 将立即失效</li>
          <li>请在安全的环境下查看和复制 Token</li>
          <li>如果怀疑 Token 泄露，请立即重新生成或更新</li>
          <li>建议使用系统生成的 Token 以获得最高安全性</li>
        </ul>
      </div>
    </div>
  )
}
