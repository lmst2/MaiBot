import type { FieldHookComponent } from '@/lib/field-hooks'
import { ExpressionSection } from '../sections/ExpressionSection'

/**
 * ExpressionSection as a Field Hook Component
 * This component replaces the entire 'expression' nested config section rendering
 */
export const ExpressionSectionHook: FieldHookComponent = ({ value, onChange }) => {
  return (
    <ExpressionSection
      config={value as any}
      onChange={(newConfig) => onChange?.(newConfig)}
    />
  )
}
