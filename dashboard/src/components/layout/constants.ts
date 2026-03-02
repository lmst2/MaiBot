import { Activity, Boxes, Database, FileSearch, FileText, Hash, Home, LayoutGrid, MessageSquare, Network, Package, Server, Settings, Sliders, Smile, UserCircle } from 'lucide-react'

import type { MenuSection } from './types'

export const menuSections: MenuSection[] = [
  {
    title: '概览',
    items: [
      { icon: Home, label: '首页', path: '/' },
    ],
  },
  {
    title: '麦麦配置编辑',
    items: [
      { icon: FileText, label: '麦麦主程序配置', path: '/config/bot' },
      { icon: Server, label: 'AI模型厂商配置', path: '/config/modelProvider', tourId: 'sidebar-model-provider' },
      { icon: Boxes, label: '模型管理与分配', path: '/config/model', tourId: 'sidebar-model-management' },
      { icon: Sliders, label: '麦麦适配器配置', path: '/config/adapter' },
    ],
  },
  {
    title: '麦麦资源管理',
    items: [
      { icon: Smile, label: '表情包管理', path: '/resource/emoji' },
      { icon: MessageSquare, label: '表达方式管理', path: '/resource/expression' },
      { icon: Hash, label: '黑话管理', path: '/resource/jargon' },
      { icon: UserCircle, label: '人物信息管理', path: '/resource/person' },
      { icon: Network, label: '知识库图谱可视化', path: '/resource/knowledge-graph' },
      { icon: Database, label: '麦麦知识库管理', path: '/resource/knowledge-base' },
    ],
  },
  {
    title: '扩展与监控',
    items: [
      { icon: Package, label: '插件市场', path: '/plugins' },
      { icon: LayoutGrid, label: '配置模板市场', path: '/config/pack-market' },
      { icon: Sliders, label: '插件配置', path: '/plugin-config' },
      { icon: FileSearch, label: '日志查看器', path: '/logs' },
      { icon: Activity, label: '计划器&回复器监控', path: '/planner-monitor' },
      { icon: MessageSquare, label: '本地聊天室', path: '/chat' },
    ],
  },
  {
    title: '系统',
    items: [
      { icon: Settings, label: '系统设置', path: '/settings' },
    ],
  },
]
