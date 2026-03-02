import { app, BrowserWindow, protocol } from 'electron'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

let mainWindow: BrowserWindow | null = null

/**
 * Register app:// custom protocol BEFORE app.whenReady()
 * This is critical for electron-vite to work correctly
 */
function registerAppProtocol() {
  protocol.registerSchemesAsPrivileged([
    {
      scheme: 'app',
      privileges: {
        secure: true,
        standard: true,
        allowServiceWorkers: true,
      },
    },
  ])
}

/**
 * Create the main application window
 */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  // Load the app using app:// protocol
  // electron-vite will handle serving the renderer from app://host/index.html
  if (process.env.VITE_DEV_SERVER_URL) {
    // Development: load from electron-vite dev server
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL)
  } else {
    // Production: load from bundled renderer
    mainWindow.loadURL('app://host/index.html')
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

/**
 * Register app:// protocol handler (for production)
 */
function registerAppProtocolHandler() {
  protocol.handle('app', (request) => {
    const filePath = new URL(request.url).pathname
    return new Response(
      `Cannot handle app:// requests. Renderer should be served by electron-vite.`
    )
  })
}

/**
 * App event: when app is ready
 */
app.on('ready', () => {
  registerAppProtocolHandler()
  createWindow()
})

/**
 * App event: when all windows are closed (non-macOS behavior)
 */
app.on('window-all-closed', () => {
  // On macOS, applications typically stay open until the user quits
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

/**
 * App event: when app is activated (macOS)
 */
app.on('activate', () => {
  if (mainWindow === null) {
    createWindow()
  }
})

// Register protocol BEFORE app.whenReady()
registerAppProtocol()
