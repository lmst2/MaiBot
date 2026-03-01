import { useState, useMemo, useRef, useCallback } from 'react'
import { AlertTriangle, Download, RotateCcw, Trash2, Upload } from 'lucide-react'

import { useAnimation } from '@/hooks/use-animation'
import { useTheme } from '@/components/use-theme'
import { useToast } from '@/hooks/use-toast'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { getComputedTokens } from '@/lib/theme/pipeline'
import { hexToHSL } from '@/lib/theme/palette'
import { defaultBackgroundConfig, defaultBackgroundEffects, defaultLightTokens } from '@/lib/theme/tokens'
import { exportThemeJSON, importThemeJSON } from '@/lib/theme/storage'
import type { BackgroundConfigMap, BackgroundEffects, ThemeTokens } from '@/lib/theme/tokens'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { CodeEditor } from '@/components/CodeEditor'
import { BackgroundEffectsControls } from '@/components/background-effects-controls'
import { BackgroundUploader } from '@/components/background-uploader'
import { ComponentCSSEditor } from '@/components/component-css-editor'
import { sanitizeCSS } from '@/lib/theme/sanitizer'
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs'

import { ThemeOption } from './ThemeOption'
import { hslToHex } from './types'


/**
 * 安全访问 tokenOverrides 中的子属性值
 * @param overrides - Partial<ThemeTokens>
 * @param section - 如 'typography', 'visual', 'layout', 'animation'
 * @param key - token 键名，如 'font-family-base'
 * @param defaultValue - 默认值
 */
function getTokenValue<T>(
  overrides: Partial<ThemeTokens> | undefined,
  section: keyof ThemeTokens,
  key: string,
  defaultValue: T
): T {
  if (!overrides || !overrides[section]) return defaultValue
  const sectionTokens = overrides[section] as Record<string, unknown> | undefined
  if (!sectionTokens || !(key in sectionTokens)) return defaultValue
  return (sectionTokens[key] ?? defaultValue) as T
}
export function AppearanceTab() {
  const { theme, setTheme, themeConfig, updateThemeConfig, resolvedTheme, resetTheme } = useTheme()
  const { enableAnimations, setEnableAnimations, enableWavesBackground, setEnableWavesBackground } = useAnimation()
  const { toast } = useToast()
  
  const [localCSS, setLocalCSS] = useState(themeConfig.customCSS || '')
  const [cssWarnings, setCssWarnings] = useState<string[]>([])
  const cssDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const bgDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const updateTokenSection = useCallback(
    <K extends keyof ThemeTokens>(section: K, partial: Partial<ThemeTokens[K]>) => {
      updateThemeConfig({
        tokenOverrides: {
          ...themeConfig.tokenOverrides,
          [section]: {
            ...defaultLightTokens[section],
            ...themeConfig.tokenOverrides?.[section],
            ...partial,
          } as ThemeTokens[K],
        },
      })
    },
    [themeConfig.tokenOverrides, updateThemeConfig]
  )

  const resetTokenSection = useCallback(
    (section: keyof ThemeTokens) => {
      const newOverrides: Partial<ThemeTokens> = { ...themeConfig.tokenOverrides }
      delete newOverrides[section]
      updateThemeConfig({ tokenOverrides: newOverrides })
    },
    [themeConfig.tokenOverrides, updateThemeConfig]
  )

  const handleCSSChange = useCallback((val: string) => {
    setLocalCSS(val)
    const result = sanitizeCSS(val)
    setCssWarnings(result.warnings)
    
    if (cssDebounceRef.current) clearTimeout(cssDebounceRef.current)
    cssDebounceRef.current = setTimeout(() => {
      updateThemeConfig({ customCSS: val })
    }, 500)
  }, [updateThemeConfig])

  const currentAccentHex = useMemo(() => {
    if (themeConfig.accentColor) {
      return hslToHex(themeConfig.accentColor)
    }
    return '#3b82f6' // 默认蓝色
  }, [themeConfig.accentColor])

  const handleAccentColorChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const hex = e.target.value
    const hsl = hexToHSL(hex)
    updateThemeConfig({ accentColor: hsl })
  }

  const handleResetAccent = () => {
    updateThemeConfig({ accentColor: '' })
  }

  const handleExport = () => {
    const json = exportThemeJSON()
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `maibot-theme-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const json = ev.target?.result as string
      const result = importThemeJSON(json)
      if (result.success) {
        // 导入成功后需要刷新页面使配置生效（因为 ThemeProvider 需要重新读取 localStorage）
        toast({ title: '导入成功', description: '主题配置已导入，页面将自动刷新' })
        setTimeout(() => window.location.reload(), 1000)
      } else {
        toast({ title: '导入失败', description: result.errors.join('; '), variant: 'destructive' })
      }
    }
    reader.readAsText(file)
    // 重置 input，允许重复选择同一文件
    e.target.value = ''
  }

  const handleResetTheme = () => {
    resetTheme()
    setLocalCSS('')
    setCssWarnings([])
    toast({ title: '重置成功', description: '主题已重置为默认值' })
  }

  const previewTokens = useMemo(() => {
    return getComputedTokens(themeConfig, resolvedTheme === 'dark').color
  }, [themeConfig, resolvedTheme])

  const bgConfig: BackgroundConfigMap = themeConfig.backgroundConfig ?? {}

  const handleBgAssetChange = (layerId: keyof BackgroundConfigMap, assetId: string | undefined) => {
    const current = bgConfig[layerId] ?? defaultBackgroundConfig
    const newMap: BackgroundConfigMap = {
      ...bgConfig,
      [layerId]: { ...current, assetId, type: assetId ? 'image' : 'none' },
    }
    if (bgDebounceRef.current) clearTimeout(bgDebounceRef.current)
    bgDebounceRef.current = setTimeout(() => updateThemeConfig({ backgroundConfig: newMap }), 500)
  }

  const handleBgEffectsChange = (layerId: keyof BackgroundConfigMap, effects: BackgroundEffects) => {
    const current = bgConfig[layerId] ?? defaultBackgroundConfig
    const newMap: BackgroundConfigMap = { ...bgConfig, [layerId]: { ...current, effects } }
    if (bgDebounceRef.current) clearTimeout(bgDebounceRef.current)
    bgDebounceRef.current = setTimeout(() => updateThemeConfig({ backgroundConfig: newMap }), 500)
  }

  const handleBgCSSChange = (layerId: keyof BackgroundConfigMap, css: string) => {
    const current = bgConfig[layerId] ?? defaultBackgroundConfig
    const newMap: BackgroundConfigMap = { ...bgConfig, [layerId]: { ...current, customCSS: css } }
    if (bgDebounceRef.current) clearTimeout(bgDebounceRef.current)
    bgDebounceRef.current = setTimeout(() => updateThemeConfig({ backgroundConfig: newMap }), 500)
  }

  const handleBgInheritChange = (layerId: keyof BackgroundConfigMap, inherit: boolean) => {
    const current = bgConfig[layerId] ?? defaultBackgroundConfig
    const newMap: BackgroundConfigMap = { ...bgConfig, [layerId]: { ...current, inherit } }
    if (bgDebounceRef.current) clearTimeout(bgDebounceRef.current)
    bgDebounceRef.current = setTimeout(() => updateThemeConfig({ backgroundConfig: newMap }), 500)
  }

  return (
    <div className="space-y-6 sm:space-y-8">
      {/* 主题模式 */}
      <div>
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">主题模式</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
          <ThemeOption
            value="light"
            current={theme}
            onChange={setTheme}
            label="浅色"
            description="始终使用浅色主题"
          />
          <ThemeOption
            value="dark"
            current={theme}
            onChange={setTheme}
            label="深色"
            description="始终使用深色主题"
          />
          <ThemeOption
            value="system"
            current={theme}
            onChange={setTheme}
            label="跟随系统"
            description="根据系统设置自动切换"
          />
        </div>
      </div>

      {/* 主题色配置 */}
      <div>
        <div className="flex items-center justify-between mb-3 sm:mb-4">
          <h3 className="text-base sm:text-lg font-semibold">主题色</h3>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleResetAccent}
            disabled={!themeConfig.accentColor}
            className="h-8"
          >
            <RotateCcw className="mr-2 h-3.5 w-3.5" />
            重置默认
          </Button>
        </div>
        
        <div className="space-y-6">
          {/* 颜色选择器 */}
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center p-4 rounded-lg border bg-card">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full border-2 border-border overflow-hidden relative shadow-sm">
                <input
                  type="color"
                  value={currentAccentHex}
                  onChange={handleAccentColorChange}
                  className="absolute inset-0 w-[150%] h-[150%] -top-1/4 -left-1/4 cursor-pointer p-0 border-0"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="accent-color-input" className="font-medium">主色调</Label>
                <p className="text-xs text-muted-foreground">点击色环选择或输入 HEX 值</p>
              </div>
            </div>
            
            <div className="flex-1 w-full sm:w-auto flex items-center gap-2">
              <Input
                id="accent-color-input"
                type="text"
                value={currentAccentHex}
                onChange={handleAccentColorChange}
                className="font-mono uppercase w-32"
                maxLength={7}
              />
            </div>
          </div>

          {/* 实时色板预览 */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-muted-foreground">实时色板预览</h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-3">
              <ColorTokenPreview name="primary" value={previewTokens.primary} foreground={previewTokens['primary-foreground']} />
              <ColorTokenPreview name="secondary" value={previewTokens.secondary} foreground={previewTokens['secondary-foreground']} />
              <ColorTokenPreview name="muted" value={previewTokens.muted} foreground={previewTokens['muted-foreground']} />
              <ColorTokenPreview name="accent" value={previewTokens.accent} foreground={previewTokens['accent-foreground']} />
              <ColorTokenPreview name="destructive" value={previewTokens.destructive} foreground={previewTokens['destructive-foreground']} />
              <ColorTokenPreview name="background" value={previewTokens.background} foreground={previewTokens.foreground} border />
              <ColorTokenPreview name="card" value={previewTokens.card} foreground={previewTokens['card-foreground']} border />
              <ColorTokenPreview name="border" value={previewTokens.border} />
            </div>
          </div>
        </div>
      </div>

      {/* 样式微调 */}
      <div>
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">界面样式微调</h3>


        <Accordion type="single" collapsible className="w-full">
          {/* 1. 字体排版 (Typography) */}
          <AccordionItem value="typography">
            <AccordionTrigger>字体排版 (Typography)</AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                <div className="flex justify-end">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => resetTokenSection('typography')}
                    disabled={!themeConfig.tokenOverrides?.typography}
                    className="h-8 text-xs"
                  >
                    <RotateCcw className="mr-2 h-3.5 w-3.5" />
                    重置默认
                  </Button>
                </div>

                <div className="space-y-2">
                  <Label>字体族 (Font Family)</Label>
                  <Select
                    value={(() => {
                      const fontFamily = getTokenValue(themeConfig.tokenOverrides, 'typography', 'font-family-base', '')
                      if (fontFamily.includes('ui-serif')) return 'serif'
                      if (fontFamily.includes('ui-monospace')) return 'mono'
                      if (fontFamily) return 'sans'
                      return 'system'
                    })()}
                    onValueChange={(val) => {
                      let fontVal = defaultLightTokens.typography['font-family-base']
                      if (val === 'serif') fontVal = 'ui-serif, Georgia, Cambria, "Times New Roman", Times, serif'
                      else if (val === 'mono') fontVal = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
                      else if (val === 'sans') fontVal = 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
                      
                      updateTokenSection('typography', {
                        'font-family-base': fontVal,
                      })
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="选择字体族" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="system">系统默认 (System)</SelectItem>
                      <SelectItem value="sans">无衬线 (Sans-serif)</SelectItem>
                      <SelectItem value="serif">衬线 (Serif)</SelectItem>
                      <SelectItem value="mono">等宽 (Monospace)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between">
                    <Label>基准字体大小 (Base Size)</Label>
                    <span className="text-sm text-muted-foreground">
                      {parseFloat(getTokenValue(themeConfig.tokenOverrides, 'typography', 'font-size-base', '1')) * 16}px
                    </span>
                  </div>
                  <Slider
                    defaultValue={[16]}
                    value={[parseFloat(getTokenValue(themeConfig.tokenOverrides, 'typography', 'font-size-base', '1')) * 16]}
                    min={12}
                    max={20}
                    step={1}
                    onValueChange={(vals) => {
                      updateTokenSection('typography', {
                        'font-size-base': `${vals[0] / 16}rem`,
                      })
                    }}
                  />
                </div>

                <div className="space-y-2">
                  <Label>行高 (Line Height)</Label>
                  <Select
                    value={String(getTokenValue(themeConfig.tokenOverrides, 'typography', 'line-height-normal', 1.5))}
                    onValueChange={(val) => {
                      updateTokenSection('typography', {
                        'line-height-normal': parseFloat(val),
                      })
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="选择行高" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1.2">紧凑 (1.2)</SelectItem>
                      <SelectItem value="1.5">正常 (1.5)</SelectItem>
                      <SelectItem value="1.75">宽松 (1.75)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* 2. 视觉效果 (Visual) */}
          <AccordionItem value="visual">
            <AccordionTrigger>视觉效果 (Visual)</AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                <div className="flex justify-end">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => resetTokenSection('visual')}
                    disabled={!themeConfig.tokenOverrides?.visual}
                    className="h-8 text-xs"
                  >
                    <RotateCcw className="mr-2 h-3.5 w-3.5" />
                    重置默认
                  </Button>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between">
                    <Label>圆角大小 (Radius)</Label>
                    <span className="text-sm text-muted-foreground">
                      {Math.round(parseFloat(getTokenValue(themeConfig.tokenOverrides, 'visual', 'radius-md', '0.375')) * 16)}px
                    </span>
                  </div>
                  <Slider
                    defaultValue={[6]}
                    value={[Math.round(parseFloat(getTokenValue(themeConfig.tokenOverrides, 'visual', 'radius-md', '0.375')) * 16)]}
                    min={0}
                    max={24}
                    step={1}
                    onValueChange={(vals) => {
                      updateTokenSection('visual', {
                        'radius-md': `${vals[0] / 16}rem`,
                      })
                    }}
                  />
                </div>

                <div className="space-y-2">
                  <Label>阴影强度 (Shadow)</Label>
                  <Select
                    value={(() => {
                      const shadowMd = String(getTokenValue(themeConfig.tokenOverrides, 'visual', 'shadow-md', ''))
                      if (shadowMd === 'none') return 'none'
                      if (shadowMd === defaultLightTokens.visual['shadow-sm']) return 'sm'
                      if (shadowMd === defaultLightTokens.visual['shadow-lg']) return 'lg'
                      if (shadowMd === defaultLightTokens.visual['shadow-xl']) return 'xl'
                      return 'md'
                    })()}
                    onValueChange={(val) => {
                      let shadowVal = defaultLightTokens.visual['shadow-md']
                      if (val === 'none') shadowVal = 'none'
                      else if (val === 'sm') shadowVal = defaultLightTokens.visual['shadow-sm']
                      else if (val === 'lg') shadowVal = defaultLightTokens.visual['shadow-lg']
                      else if (val === 'xl') shadowVal = defaultLightTokens.visual['shadow-xl']
                      
                      updateTokenSection('visual', {
                        'shadow-md': shadowVal,
                      })
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="选择阴影强度" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">无阴影 (None)</SelectItem>
                      <SelectItem value="sm">轻微 (Small)</SelectItem>
                      <SelectItem value="md">中等 (Medium)</SelectItem>
                      <SelectItem value="lg">强烈 (Large)</SelectItem>
                      <SelectItem value="xl">极强 (Extra Large)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="blur-switch">模糊效果 (Blur)</Label>
                  <Switch
                    id="blur-switch"
                    checked={getTokenValue(themeConfig.tokenOverrides, 'visual', 'blur-md', '0px') !== '0px'}
                    onCheckedChange={(checked) => {
                      updateTokenSection('visual', {
                        'blur-md': checked ? defaultLightTokens.visual['blur-md'] : '0px',
                      })
                    }}
                  />
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* 3. 布局 (Layout) */}
          <AccordionItem value="layout">
            <AccordionTrigger>布局 (Layout)</AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                <div className="flex justify-end">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => resetTokenSection('layout')}
                    disabled={!themeConfig.tokenOverrides?.layout}
                    className="h-8 text-xs"
                  >
                    <RotateCcw className="mr-2 h-3.5 w-3.5" />
                    重置默认
                  </Button>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between">
                    <Label>侧边栏宽度 (Sidebar Width)</Label>
                    <span className="text-sm text-muted-foreground">
                      {getTokenValue(themeConfig.tokenOverrides, 'layout', 'sidebar-width', '16rem')}
                    </span>
                  </div>
                  <Slider
                    defaultValue={[16]}
                    value={[parseFloat(getTokenValue(themeConfig.tokenOverrides, 'layout', 'sidebar-width', '16'))]}
                    min={12}
                    max={24}
                    step={0.5}
                    onValueChange={(vals) => {
                      updateTokenSection('layout', {
                        'sidebar-width': `${vals[0]}rem`,
                      })
                    }}
                  />
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between">
                    <Label>内容区最大宽度 (Max Width)</Label>
                    <span className="text-sm text-muted-foreground">
                      {getTokenValue(themeConfig.tokenOverrides, 'layout', 'max-content-width', '1280px')}
                    </span>
                  </div>
                  <Slider
                    defaultValue={[1280]}
                    value={[parseFloat(getTokenValue(themeConfig.tokenOverrides, 'layout', 'max-content-width', '1280').replace('px', ''))]}
                    min={960}
                    max={1600}
                    step={10}
                    onValueChange={(vals) => {
                      updateTokenSection('layout', {
                        'max-content-width': `${vals[0]}px`,
                      })
                    }}
                  />
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between">
                    <Label>基准间距 (Spacing Unit)</Label>
                    <span className="text-sm text-muted-foreground">
                      {getTokenValue(themeConfig.tokenOverrides, 'layout', 'space-unit', '0.25rem')}
                    </span>
                  </div>
                  <Slider
                    defaultValue={[0.25]}
                    value={[parseFloat(getTokenValue(themeConfig.tokenOverrides, 'layout', 'space-unit', '0.25').replace('rem', ''))]}
                    min={0.2}
                    max={0.4}
                    step={0.01}
                    onValueChange={(vals) => {
                      updateTokenSection('layout', {
                        'space-unit': `${vals[0]}rem`,
                      })
                    }}
                  />
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* 4. 动画 (Animation) */}
          <AccordionItem value="animation">
            <AccordionTrigger>动画 (Animation)</AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                <div className="flex justify-end">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => resetTokenSection('animation')}
                    disabled={!themeConfig.tokenOverrides?.animation}
                    className="h-8 text-xs"
                  >
                    <RotateCcw className="mr-2 h-3.5 w-3.5" />
                    重置默认
                  </Button>
                </div>

                <div className="space-y-2">
                  <Label>动画速度 (Speed)</Label>
                  <Select
                    value={(() => {
                      const duration = String(getTokenValue(themeConfig.tokenOverrides, 'animation', 'anim-duration-normal', '300ms'))
                      if (duration === '100ms') return 'fast'
                      if (duration === '500ms') return 'slow'
                      if (duration === '0ms') return 'off'
                      return 'normal'
                    })()}
                    onValueChange={(val) => {
                      let duration = '300ms'
                      if (val === 'fast') duration = '100ms'
                      else if (val === 'slow') duration = '500ms'
                      else if (val === 'off') duration = '0ms'
                      
                      // 如果用户选了关闭，我们也应该同步更新 enableAnimations 开关
                      if (val === 'off' && enableAnimations) {
                        setEnableAnimations(false)
                      } else if (val !== 'off' && !enableAnimations) {
                        setEnableAnimations(true)
                      }

                      updateTokenSection('animation', {
                        'anim-duration-normal': duration,
                      })
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="选择动画速度" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fast">快速 (100ms)</SelectItem>
                      <SelectItem value="normal">正常 (300ms)</SelectItem>
                      <SelectItem value="slow">慢速 (500ms)</SelectItem>
                      <SelectItem value="off">关闭 (0ms)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* 5. 背景设置 (Backgrounds) */}
          <AccordionItem value="backgrounds">
            <AccordionTrigger>背景设置 (Backgrounds)</AccordionTrigger>
            <AccordionContent>
              <div className="pt-2">
                <Tabs defaultValue="page">
                  <TabsList className="w-full grid grid-cols-5">
                    <TabsTrigger value="page">页面</TabsTrigger>
                    <TabsTrigger value="sidebar">侧边栏</TabsTrigger>
                    <TabsTrigger value="header">Header</TabsTrigger>
                    <TabsTrigger value="card">Card</TabsTrigger>
                    <TabsTrigger value="dialog">Dialog</TabsTrigger>
                  </TabsList>

                  {(['page', 'sidebar', 'header', 'card', 'dialog'] as const).map((layerId) => (
                    <TabsContent key={layerId} value={layerId} className="space-y-4 mt-4">
                      {layerId !== 'page' && (
                        <div className="flex items-center justify-between rounded-lg border bg-muted/30 px-4 py-3">
                          <div className="space-y-0.5">
                            <Label className="text-sm font-medium">继承上级背景</Label>
                            <p className="text-xs text-muted-foreground">开启后将使用上级层级的背景配置</p>
                          </div>
                          <Switch
                            checked={bgConfig[layerId]?.inherit ?? false}
                            onCheckedChange={(v) => handleBgInheritChange(layerId, v)}
                          />
                        </div>
                      )}
                      <BackgroundUploader
                        assetId={bgConfig[layerId]?.assetId}
                        onAssetSelect={(id) => handleBgAssetChange(layerId, id)}
                      />
                      <BackgroundEffectsControls
                        effects={bgConfig[layerId]?.effects ?? defaultBackgroundEffects}
                        onChange={(effects) => handleBgEffectsChange(layerId, effects)}
                      />
                      <ComponentCSSEditor
                        componentId={layerId}
                        value={bgConfig[layerId]?.customCSS ?? ''}
                        onChange={(css) => handleBgCSSChange(layerId, css)}
                      />
                    </TabsContent>
                  ))}
                </Tabs>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3 sm:mb-4">
          <div>
            <h3 className="text-base sm:text-lg font-semibold">自定义 CSS</h3>
            <p className="text-sm text-muted-foreground mt-1">
              编写自定义 CSS 来进一步个性化界面。危险的 CSS（如 @import、url()）将被自动过滤。
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setLocalCSS('')
              updateThemeConfig({ customCSS: '' })
              setCssWarnings([])
            }}
            disabled={!themeConfig.customCSS}
          >
            <Trash2 className="h-4 w-4 mr-1" />
            清除
          </Button>
        </div>
        
        <div className="rounded-lg border bg-card p-3 sm:p-4 space-y-3">
          <CodeEditor
            value={localCSS}
            language="css"
            height="250px"
            placeholder={`/* 在这里输入自定义 CSS */\n\n/* 例如: */\n/* .sidebar { background: #1a1a2e; } */`}
            onChange={handleCSSChange}
          />
          
          {cssWarnings.length > 0 && (
            <div className="rounded-md bg-yellow-50 dark:bg-yellow-950/30 border border-yellow-200 dark:border-yellow-800 p-3">
              <div className="flex items-center gap-2 text-yellow-800 dark:text-yellow-200 text-sm font-medium mb-1">
                <AlertTriangle className="h-4 w-4" />
                以下内容已被安全过滤：
              </div>
              <ul className="text-xs text-yellow-700 dark:text-yellow-300 space-y-0.5 ml-6 list-disc">
                {cssWarnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* 动效设置 */}
      <div>
        <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">动画效果</h3>
        <div className="space-y-2 sm:space-y-3">
          {/* 全局动画开关 */}
          <div className="rounded-lg border bg-card p-3 sm:p-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5 flex-1">
                <Label htmlFor="animations" className="text-base font-medium cursor-pointer">
                  启用动画效果
                </Label>
                <p className="text-sm text-muted-foreground">
                  关闭后将禁用所有过渡动画和特效，提升性能
                </p>
              </div>
              <Switch
                id="animations"
                checked={enableAnimations}
                onCheckedChange={setEnableAnimations}
              />
            </div>
          </div>

          {/* 波浪背景开关 */}
          <div className="rounded-lg border bg-card p-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5 flex-1">
                <Label htmlFor="waves-background" className="text-base font-medium cursor-pointer">
                  登录页波浪背景
                </Label>
                <p className="text-sm text-muted-foreground">
                  关闭后登录页将使用纯色背景，适合低性能设备
                </p>
              </div>
              <Switch
                id="waves-background"
                checked={enableWavesBackground}
                onCheckedChange={setEnableWavesBackground}
              />
            </div>
           </div>
         </div>
       </div>

       {/* 主题导入/导出 */}
       <div>
         <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">主题导入/导出</h3>
         <div className="rounded-lg border bg-card p-3 sm:p-4 space-y-3">
           <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
             {/* 导出按钮 */}
             <Button 
               onClick={handleExport}
               variant="outline"
               className="gap-2"
             >
               <Download className="h-4 w-4" />
               导出主题
             </Button>

             {/* 导入按钮 */}
             <Button 
               onClick={() => fileInputRef.current?.click()}
               variant="outline"
               className="gap-2"
             >
               <Upload className="h-4 w-4" />
               导入主题
             </Button>

             {/* 重置按钮 */}
             <AlertDialog>
               <AlertDialogTrigger asChild>
                 <Button 
                   variant="outline"
                   className="gap-2"
                 >
                   <RotateCcw className="h-4 w-4" />
                   重置为默认
                 </Button>
               </AlertDialogTrigger>
               <AlertDialogContent>
                 <AlertDialogHeader>
                   <AlertDialogTitle>确认重置主题</AlertDialogTitle>
                   <AlertDialogDescription>
                     这将重置所有主题设置为默认值，包括颜色、字体、布局和自定义 CSS。此操作不可撤销，确定要继续吗？
                   </AlertDialogDescription>
                 </AlertDialogHeader>
                 <AlertDialogFooter>
                   <AlertDialogCancel>取消</AlertDialogCancel>
                   <AlertDialogAction onClick={handleResetTheme}>
                     确认重置
                   </AlertDialogAction>
                 </AlertDialogFooter>
               </AlertDialogContent>
             </AlertDialog>
           </div>

           {/* 隐藏的文件输入 */}
           <input
             ref={fileInputRef}
             type="file"
             accept=".json"
             onChange={handleImport}
             className="hidden"
           />

           <p className="text-xs text-muted-foreground">
             导出主题为 JSON 文件便于分享或备份，导入时会自动应用所有配置。
           </p>
         </div>
       </div>
     </div>
  )
}

function ColorTokenPreview({ name, value, foreground, border }: { name: string, value: string, foreground?: string, border?: boolean }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div 
        className={cn("h-16 rounded-md shadow-sm flex items-center justify-center text-xs font-medium", border && "border border-border")}
        style={{ backgroundColor: `hsl(${value})`, color: foreground ? `hsl(${foreground})` : undefined }}
      >
        Aa
      </div>
      <div className="text-xs text-muted-foreground text-center truncate" title={name}>
        {name}
      </div>
    </div>
  )
}
