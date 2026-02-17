import type { FieldHookComponent } from '@/lib/field-hooks'
import { DebugSection } from '../sections/DebugSection'

/**
 * DebugSection as a Field Hook Component
 * This component replaces the entire 'debug' nested config section rendering
 */
export const DebugSectionHook: FieldHookComponent = ({ value, onChange }) => {
  return (
    <DebugSection
      config={value as any}
      onChange={(newConfig) => onChange?.(newConfig)}
    />
  )
}
