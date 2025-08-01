# DataMax 项目文档维护手册

> 本文档是 DataMax 项目官方文档网站的内部维护手册。网站基于 [Docusaurus](https://docusaurus.io/) 构建。

**在线访问地址：[https://Hi-Dolphi.github.io/datamax/](https://Hi-Dolphi.github.io/datamax/)**

## ✨ 核心特性

* **内容丰富**: 涵盖从快速入门到高级功能的所有内容。
* **全文搜索**: 由 Algolia 提供支持的快速全文搜索功能。
* **版本控制**: 支持多版本文档管理。
* **明暗模式**: 自动适应系统的外观偏好。

---

## 🚀 本地开发 (Local Development)

团队成员在修改或添加任何内容之前，都需要在本地将网站运行起来，以便实时预览效果。

### 1. 环境准备 (Prerequisites)

* [Node.js](https://nodejs.org/en/) (版本 >= 18.0)
* [NPM](https://www.npmjs.com/) (随 Node.js 自动安装)

你可以通过在终端运行以下命令来检查版本：
```bash
node -v
npm -v
```

### 2. 安装与运行 (Installation & Running)

1.  克隆本仓库到本地。

2.  进入 Docusaurus 项目目录 (网站代码位于 `datamax_docs` 子目录中)：
    ```bash
    cd datamax/datamax_docs
    ```
    **注意：后续所有命令都需要在此目录下执行。**

3.  安装项目依赖 (首次运行时或依赖更新后执行)：
    ```bash
    npm install
    ```

4.  启动本地开发服务器：
    ```bash
    npm run start
    ```
    执行后，网站将在 `http://localhost:3000` 上运行。此服务支持热重载，当你修改文件并保存后，浏览器会自动刷新，方便实时预览。

---

## 📝 内容维护指南 (Content Maintenance Guide)

本节为团队成员提供详细的日常维护指引。

### 项目结构概览

与我们日常工作最相关的文件和文件夹都位于 `datamax_docs` 内：

```
datamax_docs/
├── docs/                  <-- ✅ 存放所有文档 Markdown 文件的地方
├── static/
│   └── img/               <-- ✅ 存放图片等静态资源
├── src/
│   └── css/
│       └── custom.css     <-- ✅ 自定义网站样式
├── sidebars.js            <-- ❗️ 侧边栏结构定义文件
└── docusaurus.config.js   <-- ❗️ 核心配置文件 (网站标题、导航栏等)
```

### 修改与新增文档

* **修改文档**：直接在 `docs/` 文件夹中找到对应的 Markdown 文件，使用任何文本编辑器（推荐 VS Code）打开并修改即可。
* **新增文档**：在 `docs/` 文件夹或其子文件夹中创建一个新的 `.md` 文件。

### 侧边栏如何工作？(自动生成)

我们的侧边栏配置为**自动生成模式** (`autogenerated`)。

这意味着：
* **您不需要手动编辑 `sidebars.js` 文件。**
* 只需在 `docs` 文件夹中**添加、删除或重命名** Markdown 文件或文件夹，侧边栏就会**自动更新**。
* 侧边栏的顺序和分组结构会**自动映射 `docs` 目录的文件和文件夹结构**，通常按字母顺序排列。

---

## 🎨 高级配置 (Advanced Configuration)

需要修改网站外观或整体结构时，请编辑以下文件：

* **网站元数据** (标题、Logo、导航栏等): `docusaurus.config.js`
* **网站主题颜色**: `src/css/custom.css`

修改这些文件需要对 Docusaurus 有一定了解，请谨慎操作或参考官方文档。

---

## 部署 (Deployment)

本网站通过 **GitHub Actions** 自动部署。任何推送到 `main` 分支的提交都会触发自动构建和发布流程，将最新的文档部署到 GitHub Pages。

**团队成员只需将修改推送到 `main` 分支即可，无需执行任何手动部署命令。**

***

*该文档由 Docusaurus 强力驱动。*