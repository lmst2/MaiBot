import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  main: {
    entry: 'electron/main/index.ts',
    vite: {
      build: {
        rollupOptions: {
          external: ['electron'],
        },
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, './src'),
        },
      },
    },
  },
  preload: {
    entry: 'electron/preload/index.ts',
    vite: {
      build: {
        rollupOptions: {
          external: ['electron'],
        },
      },
    },
  },
  renderer: {
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    plugins: [react()],
    server: {
      port: 7999,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8001',
          changeOrigin: true,
          ws: true,
          cookieDomainRewrite: '',
          cookiePathRewrite: '/',
        },
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            // React core
            'react-vendor': ['react', 'react-dom', 'react/jsx-runtime'],

            // TanStack Router
            router: ['@tanstack/react-router', '@tanstack/react-virtual'],

            // Radix UI core
            'radix-core': [
              '@radix-ui/react-dialog',
              '@radix-ui/react-select',
              '@radix-ui/react-checkbox',
              '@radix-ui/react-label',
              '@radix-ui/react-slot',
              '@radix-ui/react-toast',
              '@radix-ui/react-tooltip',
            ],

            // Radix UI extras
            'radix-extra': [
              '@radix-ui/react-alert-dialog',
              '@radix-ui/react-avatar',
              '@radix-ui/react-collapsible',
              '@radix-ui/react-context-menu',
              '@radix-ui/react-popover',
              '@radix-ui/react-progress',
              '@radix-ui/react-scroll-area',
              '@radix-ui/react-separator',
              '@radix-ui/react-slider',
              '@radix-ui/react-switch',
              '@radix-ui/react-tabs',
            ],

            // Icons
            icons: ['lucide-react'],

            // Charts
            charts: ['recharts'],

            // CodeMirror
            codemirror: [
              '@uiw/react-codemirror',
              '@codemirror/lang-javascript',
              '@codemirror/lang-json',
              '@codemirror/lang-python',
              '@codemirror/lint',
              '@codemirror/theme-one-dark',
            ],

            // ReactFlow
            reactflow: ['reactflow', 'dagre'],

            // Markdown
            markdown: [
              'react-markdown',
              'remark-gfm',
              'remark-math',
              'rehype-katex',
              'katex',
            ],

            // Uppy
            uppy: [
              '@uppy/core',
              '@uppy/dashboard',
              '@uppy/react',
              '@uppy/xhr-upload',
            ],

            // Drag and drop
            dnd: [
              '@dnd-kit/core',
              '@dnd-kit/sortable',
              '@dnd-kit/utilities',
            ],

            // Utils
            utils: [
              'date-fns',
              'clsx',
              'tailwind-merge',
              'class-variance-authority',
              'axios',
            ],

            // Misc
            misc: [
              'react-joyride',
              'react-day-picker',
              'cmdk',
            ],
          },
        },
      },
      chunkSizeWarningLimit: 500,
    },
  },
})
