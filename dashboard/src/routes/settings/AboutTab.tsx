import { ScrollArea } from '@/components/ui/scroll-area'

import { APP_NAME, APP_VERSION } from '@/lib/version'
import { cn } from '@/lib/utils'

import { LibraryItem } from './LibraryItem'

export function AboutTab() {
  return (
    <div className="space-y-4 sm:space-y-6">
      {/* GitHub 开源地址 */}
      <div className="rounded-lg border-2 border-primary/30 bg-gradient-to-r from-primary/5 to-primary/10 p-4 sm:p-6">
        <div className="flex items-start gap-3 sm:gap-4">
          <div className="flex-shrink-0 rounded-lg bg-primary/10 p-2 sm:p-3">
            <svg
              className="h-6 w-6 sm:h-8 sm:w-8 text-primary"
              fill="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg sm:text-xl font-bold text-foreground mb-2">
              开源项目
            </h3>
            <p className="text-sm sm:text-base text-muted-foreground mb-3">
              本项目在 GitHub 开源，欢迎 Star ⭐ 支持！
            </p>
            <a
              href="https://github.com/Mai-with-u/MaiBot-Dashboard"
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                "inline-flex items-center gap-2 px-4 py-2 rounded-lg",
                "bg-primary text-primary-foreground font-medium text-sm",
                "hover:bg-primary/90 transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              )}
            >
              <svg
                className="h-4 w-4"
                fill="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  fillRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                  clipRule="evenodd"
                />
              </svg>
              前往 GitHub
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
            </a>
          </div>
        </div>
      </div>

      {/* 应用信息 */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">关于 {APP_NAME}</h3>
        <div className="space-y-2 text-xs sm:text-sm text-muted-foreground">
          <p>版本: {APP_VERSION}</p>
          <p>麦麦（MaiBot）的现代化 Web 管理界面</p>
        </div>
      </div>

      {/* 作者信息 */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">作者</h3>
        <div className="space-y-3">
          <div className="space-y-1">
            <p className="text-sm font-medium">MaiBot 核心</p>
            <p className="text-xs sm:text-sm text-muted-foreground">Mai-with-u</p>
          </div>
          <div className="space-y-1">
            <p className="text-sm font-medium">WebUI</p>
            <p className="text-xs sm:text-sm text-muted-foreground">Mai-with-u <a href="https://github.com/DrSmoothl" target="_blank" rel="noopener noreferrer" className="text-primary underline">@MotricSeven</a></p>
          </div>
        </div>
      </div>

      {/* 技术栈 */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">技术栈</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs sm:text-sm text-muted-foreground">
          <div className="space-y-1.5">
            <p className="font-medium text-foreground">前端框架</p>
            <ul className="space-y-0.5 list-disc list-inside">
              <li>React 19.2.0</li>
              <li>TypeScript 5.7.2</li>
              <li>Vite 6.0.7</li>
              <li>TanStack Router 1.94.2</li>
            </ul>
          </div>
          <div className="space-y-1.5">
            <p className="font-medium text-foreground">UI 组件</p>
            <ul className="space-y-0.5 list-disc list-inside">
              <li>shadcn/ui</li>
              <li>Radix UI</li>
              <li>Tailwind CSS 3.4.17</li>
              <li>Lucide Icons</li>
            </ul>
          </div>
          <div className="space-y-1.5">
            <p className="font-medium text-foreground">后端</p>
            <ul className="space-y-0.5 list-disc list-inside">
              <li>Python 3.12+</li>
              <li>FastAPI</li>
              <li>Uvicorn</li>
              <li>WebSocket</li>
            </ul>
          </div>
          <div className="space-y-1.5">
            <p className="font-medium text-foreground">构建工具</p>
            <ul className="space-y-0.5 list-disc list-inside">
              <li>Bun / npm</li>
              <li>ESLint 9.17.0</li>
              <li>PostCSS</li>
            </ul>
          </div>
        </div>
      </div>

      {/* 开源感谢 */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">开源库感谢</h3>
        <p className="text-xs sm:text-sm text-muted-foreground mb-3">
          本项目使用了以下优秀的开源库，感谢他们的贡献：
        </p>
        <ScrollArea className="h-[300px] sm:h-[400px]">
          <div className="space-y-4 pr-4">
            {/* UI 框架 */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">UI 框架与组件</p>
              <div className="grid gap-2 text-xs sm:text-sm">
                <LibraryItem name="React" description="用户界面构建库" license="MIT" />
                <LibraryItem name="shadcn/ui" description="优雅的 React 组件库" license="MIT" />
                <LibraryItem name="Radix UI" description="无样式的可访问组件库" license="MIT" />
                <LibraryItem name="Tailwind CSS" description="实用优先的 CSS 框架" license="MIT" />
                <LibraryItem name="Lucide React" description="精美的图标库" license="ISC" />
              </div>
            </div>

            {/* 路由与状态 */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">路由与状态管理</p>
              <div className="grid gap-2 text-xs sm:text-sm">
                <LibraryItem name="TanStack Router" description="类型安全的路由库" license="MIT" />
                <LibraryItem name="Zustand" description="轻量级状态管理" license="MIT" />
              </div>
            </div>

            {/* 表单与验证 */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">表单处理</p>
              <div className="grid gap-2 text-xs sm:text-sm">
                <LibraryItem name="React Hook Form" description="高性能表单库" license="MIT" />
                <LibraryItem name="Zod" description="TypeScript 优先的 schema 验证" license="MIT" />
              </div>
            </div>

            {/* 工具库 */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">工具库</p>
              <div className="grid gap-2 text-xs sm:text-sm">
                <LibraryItem name="clsx" description="条件 className 构建工具" license="MIT" />
                <LibraryItem name="tailwind-merge" description="Tailwind 类名合并工具" license="MIT" />
                <LibraryItem name="class-variance-authority" description="组件变体管理" license="Apache-2.0" />
                <LibraryItem name="date-fns" description="现代化日期处理库" license="MIT" />
              </div>
            </div>

            {/* 动画 */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">动画效果</p>
              <div className="grid gap-2 text-xs sm:text-sm">
                <LibraryItem name="Framer Motion" description="React 动画库" license="MIT" />
                <LibraryItem name="vaul" description="抽屉组件动画" license="MIT" />
              </div>
            </div>

            {/* 后端相关 */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">后端框架</p>
              <div className="grid gap-2 text-xs sm:text-sm">
                <LibraryItem name="FastAPI" description="现代化 Python Web 框架" license="MIT" />
                <LibraryItem name="Uvicorn" description="ASGI 服务器" license="BSD-3-Clause" />
                <LibraryItem name="Pydantic" description="数据验证库" license="MIT" />
                <LibraryItem name="python-multipart" description="文件上传支持" license="Apache-2.0" />
              </div>
            </div>

            {/* 开发工具 */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">开发工具</p>
              <div className="grid gap-2 text-xs sm:text-sm">
                <LibraryItem name="TypeScript" description="JavaScript 的超集" license="Apache-2.0" />
                <LibraryItem name="Vite" description="下一代前端构建工具" license="MIT" />
                <LibraryItem name="ESLint" description="JavaScript 代码检查工具" license="MIT" />
                <LibraryItem name="PostCSS" description="CSS 转换工具" license="MIT" />
              </div>
            </div>
          </div>
        </ScrollArea>
      </div>

      {/* 许可证 */}
      <div className="rounded-lg border bg-card p-4 sm:p-6">
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">开源许可</h3>
        <div className="space-y-3">
          <div className="rounded-lg bg-primary/5 border border-primary/20 p-3 sm:p-4">
            <div className="flex items-start gap-2 sm:gap-3">
              <div className="flex-shrink-0 mt-0.5">
                <div className="rounded-md bg-primary/10 px-2 py-1">
                  <span className="text-xs sm:text-sm font-bold text-primary">GPLv3</span>
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm sm:text-base font-semibold text-foreground mb-1">
                  MaiBot WebUI
                </p>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  本项目采用 GNU General Public License v3.0 开源许可证。
                  您可以自由地使用、修改和分发本软件，但必须保持相同的开源许可。
                </p>
              </div>
            </div>
          </div>
          <p className="text-xs sm:text-sm text-muted-foreground">
            本项目依赖的所有开源库均遵循各自的开源许可证（MIT、Apache-2.0、BSD 等）。
            感谢所有开源贡献者的无私奉献。
          </p>
        </div>
      </div>
    </div>
  )
}
