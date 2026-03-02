import { useEffect, useState } from 'react'

import { BackgroundLayer } from '@/components/background-layer'
import { BackToTop } from '@/components/back-to-top'
import { HttpWarningBanner } from '@/components/http-warning-banner'
import { TooltipProvider } from '@/components/ui/tooltip'
import { useTheme } from '@/components/use-theme'
import { useAuthGuard } from '@/hooks/use-auth'
import { useBackground } from '@/hooks/use-background'

import { TitleBar } from '@/components/electron/TitleBar'
import { isElectron } from '@/lib/runtime'
import { cn } from '@/lib/utils'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import type { LayoutProps } from './types'

export function Layout({ children }: LayoutProps) {
  const { checking } = useAuthGuard() // 检查认证状态
  
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [tooltipsEnabled, setTooltipsEnabled] = useState(false) // 控制 tooltip 启用状态
  const { theme, setTheme } = useTheme()

  // 侧边栏状态变化时，延迟启用/禁用 tooltip
  useEffect(() => {
    if (sidebarOpen) {
      // 侧边栏展开时，立即禁用 tooltip
      setTooltipsEnabled(false)
    } else {
      // 侧边栏收起时，等待动画完成后再启用 tooltip
      const timer = setTimeout(() => {
        setTooltipsEnabled(true)
      }, 350) // 稍大于 CSS transition duration (300ms)
      return () => clearTimeout(timer)
    }
  }, [sidebarOpen])

  // 搜索快捷键监听（Cmd/Ctrl + K）
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setSearchOpen(true)
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // 认证检查中，显示加载状态
  if (checking) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="text-muted-foreground">正在验证登录状态...</div>
      </div>
    )
  }

  // 获取实际应用的主题（处理 system 情况）
  const getActualTheme = () => {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return theme
  }

  const actualTheme = getActualTheme()
  const pageBg = useBackground('page')

  return (
    <TooltipProvider delayDuration={300}>
      {isElectron() && <TitleBar />}
      <div className={cn('flex h-screen overflow-hidden', isElectron() && 'pt-8')}>
      {/* Sidebar */}
      <Sidebar
        sidebarOpen={sidebarOpen}
        mobileMenuOpen={mobileMenuOpen}
        tooltipsEnabled={tooltipsEnabled}
        onMobileMenuClose={() => setMobileMenuOpen(false)}
      />

      {/* Mobile overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* HTTP 安全警告横幅 */}
        <HttpWarningBanner />
        
        {/* Topbar */}
        <Header
          sidebarOpen={sidebarOpen}
          mobileMenuOpen={mobileMenuOpen}
          searchOpen={searchOpen}
          actualTheme={actualTheme}
          onSidebarToggle={() => setSidebarOpen(!sidebarOpen)}
          onMobileMenuToggle={() => setMobileMenuOpen(!mobileMenuOpen)}
          onSearchOpenChange={setSearchOpen}
          onThemeChange={setTheme}
        />

        {/* Page content */}
        <main className="relative flex-1 overflow-hidden bg-background">
          <BackgroundLayer config={pageBg} layerId="page" />
          {children}
        </main>

        {/* Back to Top Button */}
        <BackToTop />
      </div>
      </div>
    </TooltipProvider>
  )
}
