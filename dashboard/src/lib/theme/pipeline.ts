import type { ThemeTokens, UserThemeConfig } from './tokens'

import { generatePalette } from './palette'
import { getPresetById } from './presets'
import { sanitizeCSS } from './sanitizer'
import { defaultDarkTokens, defaultLightTokens, tokenToCSSVarName } from './tokens'

const CUSTOM_CSS_ID = 'maibot-custom-css'

const mergeTokens = (base: ThemeTokens, overrides: Partial<ThemeTokens>): ThemeTokens => {
  return {
    color: {
      ...base.color,
      ...(overrides.color ?? {}),
    },
    typography: {
      ...base.typography,
      ...(overrides.typography ?? {}),
    },
    visual: {
      ...base.visual,
      ...(overrides.visual ?? {}),
    },
    layout: {
      ...base.layout,
      ...(overrides.layout ?? {}),
    },
    animation: {
      ...base.animation,
      ...(overrides.animation ?? {}),
    },
  }
}

const buildTokens = (config: UserThemeConfig, isDark: boolean): ThemeTokens => {
  const baseTokens = isDark ? defaultDarkTokens : defaultLightTokens
  let mergedTokens = mergeTokens(baseTokens, {})

  if (config.accentColor) {
    const paletteTokens = generatePalette(config.accentColor, isDark)
    mergedTokens = mergeTokens(mergedTokens, { color: paletteTokens })
  }

  if (config.selectedPreset) {
    const preset = getPresetById(config.selectedPreset)
    if (preset?.tokens) {
      mergedTokens = mergeTokens(mergedTokens, preset.tokens)
    }
  }

  if (config.tokenOverrides) {
    mergedTokens = mergeTokens(mergedTokens, config.tokenOverrides)
  }

  return mergedTokens
}

export function getComputedTokens(config: UserThemeConfig, isDark: boolean): ThemeTokens {
  return buildTokens(config, isDark)
}

export function injectTokensAsCSS(tokens: ThemeTokens, target: HTMLElement): void {
  Object.entries(tokens.color).forEach(([key, value]) => {
    target.style.setProperty(tokenToCSSVarName('color', key), String(value))
  })

  Object.entries(tokens.typography).forEach(([key, value]) => {
    target.style.setProperty(tokenToCSSVarName('typography', key), String(value))
  })

  Object.entries(tokens.visual).forEach(([key, value]) => {
    target.style.setProperty(tokenToCSSVarName('visual', key), String(value))
  })

  Object.entries(tokens.layout).forEach(([key, value]) => {
    target.style.setProperty(tokenToCSSVarName('layout', key), String(value))
  })

  Object.entries(tokens.animation).forEach(([key, value]) => {
    target.style.setProperty(tokenToCSSVarName('animation', key), String(value))
  })
}

export function injectCustomCSS(css: string): void {
  if (css.trim().length === 0) {
    removeCustomCSS()
    return
  }

  const existing = document.getElementById(CUSTOM_CSS_ID)
  if (existing) {
    existing.textContent = css
    return
  }

  const style = document.createElement('style')
  style.id = CUSTOM_CSS_ID
  style.textContent = css
  document.head.appendChild(style)
}

export function removeCustomCSS(): void {
  const existing = document.getElementById(CUSTOM_CSS_ID)
  if (existing) {
    existing.remove()
  }
}

export function applyThemePipeline(config: UserThemeConfig, isDark: boolean): void {
  const root = document.documentElement
  const tokens = buildTokens(config, isDark)

  injectTokensAsCSS(tokens, root)

  if (config.customCSS) {
    const sanitized = sanitizeCSS(config.customCSS)
    if (sanitized.css.trim().length > 0) {
      injectCustomCSS(sanitized.css)
      return
    }
  }

  removeCustomCSS()
}
