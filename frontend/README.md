# YOLO智能检测平台 - Vue前端

基于Vue 3 + Vite + Pinia构建的现代化前端应用，用于YOLO智能检测平台的用户界面。

## 技术栈

- **Vue 3** - 渐进式JavaScript框架
- **Vite** - 下一代前端构建工具
- **Vue Router** - 官方路由管理器
- **Pinia** - 状态管理库
- **Axios** - HTTP客户端
- **Bootstrap 5** - UI框架
- **Bootstrap Icons** - 图标库

## 项目结构

```
frontend/
├── public/                 # 静态资源
├── src/
│   ├── components/         # 可复用组件
│   │   ├── NotificationContainer.vue
│   │   └── settings/      # 设置相关组件
│   │       ├── GeneralSettings.vue
│   │       ├── ModelSettings.vue
│   │       ├── DetectionSettings.vue
│   │       ├── CameraSettings.vue
│   │       ├── NotificationSettings.vue
│   │       └── SystemSettings.vue
│   ├── layouts/           # 布局组件
│   │   └── MainLayout.vue
│   ├── router/            # 路由配置
│   │   └── index.js
│   ├── stores/            # Pinia状态管理
│   │   ├── app.js
│   │   ├── detection.js
│   │   └── settings.js
│   ├── styles/            # 全局样式
│   │   └── main.css
│   ├── views/             # 页面组件
│   │   ├── Dashboard.vue
│   │   ├── LiveDetection.vue
│   │   ├── History.vue
│   │   ├── Analysis.vue
│   │   └── Settings.vue
│   ├── App.vue            # 根组件
│   └── main.js            # 入口文件
├── index.html             # HTML模板
├── package.json           # 项目依赖
├── vite.config.js         # Vite配置
└── .gitignore             # Git忽略文件
```

## 功能特性

### 1. 控制台 (Dashboard)
- 系统概览和实时状态
- 摄像头状态监控
- 实时检测趋势图表
- 最近检测记录

### 2. 实时检测 (Live Detection)
- 多摄像头视频流展示
- 支持1/2/4/6/9画面布局切换
- 实时检测统计信息
- 视频控制功能（全屏、截图、设置）
- 实时检测结果列表

### 3. 历史记录 (History)
- 历史检测记录查询
- 多维度筛选（时间、摄像头、类别）
- 分页浏览
- 数据导出功能

### 4. 数据分析 (Analysis)
- 检测趋势图表
- 类别分布分析
- 时段统计分析
- 区域分布分析
- 详细统计表格

### 5. 系统设置 (Settings)
- 常规设置（系统名称、语言、时区）
- 模型配置（模型选择、设备、阈值）
- 检测参数（跟踪、分类、分割）
- 摄像头设置（分辨率、帧率、录制）
- 通知设置（邮件、Webhook）
- 系统维护（备份、日志、数据管理）

## 安装和运行

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 开发模式

```bash
npm run dev
```

应用将在 http://localhost:3000 启动

### 3. 生产构建

```bash
npm run build
```

构建产物将输出到 `dist/` 目录

### 4. 预览构建

```bash
npm run preview
```

## API代理配置

开发环境下，前端通过Vite代理将 `/api` 请求转发到后端服务器：

```javascript
// vite.config.js
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true
    }
  }
}
```

确保后端服务运行在 `http://localhost:8000`

## 状态管理

### App Store
- 应用全局状态
- 侧边栏状态
- 通知管理

### Detection Store
- 摄像头管理
- 检测结果
- 检测控制
- 参数设置

### Settings Store
- 系统设置
- 设置持久化
- 设置重置

## 设计系统

### 颜色
- 主色: `#0066FF` (蓝色)
- 成功: `#00C853` (绿色)
- 警告: `#FFB300` (橙色)
- 错误: `#FF1744` (红色)
- 信息: `#00B0FF` (浅蓝)

### 间距
- xs: 4px
- sm: 8px
- md: 16px
- lg: 24px
- xl: 32px
- 2xl: 48px

### 圆角
- sm: 6px
- md: 8px
- lg: 12px
- xl: 16px

## 浏览器支持

- Chrome (最新版)
- Firefox (最新版)
- Safari (最新版)
- Edge (最新版)

## 开发建议

1. **组件开发**: 遵循Vue 3 Composition API规范
2. **样式管理**: 使用CSS变量保持一致性
3. **状态管理**: 复杂状态使用Pinia，简单状态使用组件内部状态
4. **API调用**: 统一使用Axios，错误处理集中管理
5. **代码规范**: 使用ESLint和Prettier保持代码质量

## 后续开发计划

- [ ] 集成图表库（ECharts/Chart.js）
- [ ] 添加用户认证和权限管理
- [ ] 实现WebSocket实时数据推送
- [ ] 添加单元测试和E2E测试
- [ ] 优化性能和打包体积
- [ ] 添加国际化支持
- [ ] 实现离线功能（PWA）
