import { BookOpen, ChevronLeft, LogOut, Menu, Moon, PieChart, Search, Sun } from 'lucide-react'
import { Link } from '@tanstack/react-router'

import { BackgroundLayer } from '@/components/background-layer'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { SearchDialog } from '@/components/search-dialog'
import { cn } from '@/lib/utils'
import { useBackground } from '@/hooks/use-background'
import { logout } from '@/lib/fetch-with-auth'
import { toggleThemeWithTransition } from '@/components/use-theme'

interface HeaderProps {
  sidebarOpen: boolean
  mobileMenuOpen: boolean
  searchOpen: boolean
  actualTheme: 'light' | 'dark'
  onSidebarToggle: () => void
  onMobileMenuToggle: () => void
  onSearchOpenChange: (open: boolean) => void
  onThemeChange: (theme: 'light' | 'dark' | 'system') => void
}

export function Header({
  sidebarOpen,

  searchOpen,
  actualTheme,
  onSidebarToggle,
  onMobileMenuToggle,
  onSearchOpenChange,
  onThemeChange,
}: HeaderProps) {
  const headerBg = useBackground('header')

  const handleLogout = async () => {
    await logout()
  }

  return (
    <header className="flex h-16 items-center justify-between border-b bg-card/80 backdrop-blur-md px-4 sticky top-0 z-10">
      <BackgroundLayer config={headerBg} layerId="header" />
      <div className="flex items-center gap-4">
        {/* 移动端菜单按钮 */}
        <button
          onClick={onMobileMenuToggle}
          className="rounded-lg p-2 hover:bg-accent lg:hidden"
        >
          <Menu className="h-5 w-5" />
        </button>
        
        {/* 桌面端侧边栏收起/展开按钮 */}
        <button
          onClick={onSidebarToggle}
          className="hidden rounded-lg p-2 hover:bg-accent lg:block"
          title={sidebarOpen ? '收起侧边栏' : '展开侧边栏'}
        >
          <ChevronLeft
            className={cn('h-5 w-5 transition-transform', !sidebarOpen && 'rotate-180')}
          />
        </button>
      </div>

      <div className="flex items-center gap-2">
        {/* 年度总结入口 */}
        <Link to="/annual-report">
          <Button
            variant="ghost"
            size="sm"
            className="gap-2 bg-gradient-to-r from-pink-500/10 to-purple-500/10 hover:from-pink-500/20 hover:to-purple-500/20 border border-pink-500/20"
            title="查看年度总结"
          >
            <PieChart className="h-4 w-4 text-pink-500" />
            <span className="hidden sm:inline bg-gradient-to-r from-pink-500 to-purple-500 bg-clip-text text-transparent font-medium">2025 年度总结</span>
          </Button>
        </Link>

        {/* 搜索框 */}
        <button
          onClick={() => onSearchOpenChange(true)}
          className="relative hidden md:flex items-center w-64 h-9 pl-9 pr-16 bg-background/50 border rounded-md hover:bg-accent/50 transition-colors text-left"
        >
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">搜索...</span>
          <Kbd size="sm" className="absolute right-2 top-1/2 -translate-y-1/2">
            <span className="text-xs">⌘</span>K
          </Kbd>
        </button>

        {/* 搜索对话框 */}
        <SearchDialog open={searchOpen} onOpenChange={onSearchOpenChange} />

        {/* 麦麦文档链接 */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => window.open('https://docs.mai-mai.org', '_blank')}
          className="gap-2"
          title="查看麦麦文档"
        >
          <BookOpen className="h-4 w-4" />
          <span className="hidden sm:inline">麦麦文档</span>
        </Button>

        {/* 主题切换按钮 */}
        <button
          onClick={(e) => {
            const newTheme = actualTheme === 'dark' ? 'light' : 'dark'
            toggleThemeWithTransition(newTheme, onThemeChange, e)
          }}
          className="rounded-lg p-2 hover:bg-accent"
          title={actualTheme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
        >
          {actualTheme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </button>

        {/* 分隔线 */}
        <div className="h-6 w-px bg-border" />

        {/* 登出按钮 */}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleLogout}
          className="gap-2"
          title="登出系统"
        >
          <LogOut className="h-4 w-4" />
          <span className="hidden sm:inline">登出</span>
        </Button>
      </div>
    </header>
  )
}
