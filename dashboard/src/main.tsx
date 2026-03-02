import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from '@tanstack/react-router'
import './index.css'
import { router } from './router'
import { AssetStoreProvider } from './components/asset-provider'
import { ThemeProvider } from './components/theme-provider'
import { AnimationProvider } from './components/animation-provider'
import { TourProvider, TourRenderer } from './components/tour'
import { Toaster } from './components/ui/toaster'
import { ErrorBoundary } from './components/error-boundary'
import { BackendSetupWizard } from './components/electron/BackendSetupWizard'
import { isElectron } from './lib/runtime'

function ElectronShell() {
  const [isFirstLaunch, setIsFirstLaunch] = useState(false)

  useEffect(() => {
    window.electronAPI!.isFirstLaunch().then(setIsFirstLaunch)
  }, [])

  return <BackendSetupWizard open={isFirstLaunch} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <AssetStoreProvider>
        <ThemeProvider defaultTheme="system">
          <AnimationProvider>
            <TourProvider>
              {isElectron() && <ElectronShell />}
              <RouterProvider router={router} />
              <TourRenderer />
              <Toaster />
            </TourProvider>
          </AnimationProvider>
        </ThemeProvider>
      </AssetStoreProvider>
    </ErrorBoundary>
  </StrictMode>
)
