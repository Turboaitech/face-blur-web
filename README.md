# 人脸模糊工具 | Face Blur Tool

在线人脸模糊工具，纯浏览器端处理，图片不会上传至任何服务器。

## 功能

- 🔍 自动人脸检测 (face-api.js / SSD MobileNet)
- 🌫 三种模糊模式：高斯模糊 / 像素马赛克 / 纯黑遮挡
- 🎛 可调参数：模糊强度、扩展范围
- ✅ 支持选择性模糊（多张人脸可单独开关）
- 📋 支持粘贴截图 (Ctrl+V)
- 📱 响应式布局，支持移动端
- 🔒 纯前端，零数据上传

## 本地开发

```bash
npm install
npm run dev
```

访问 http://localhost:3000

## 部署到 Vercel

### 方法一：命令行部署

```bash
# 安装 Vercel CLI
npm i -g vercel

# 在项目目录下运行
vercel

# 部署到生产环境
vercel --prod
```

### 方法二：GitHub 集成

1. 将代码推送到 GitHub 仓库
2. 访问 https://vercel.com/new
3. 导入你的 GitHub 仓库
4. 点击 Deploy，完成！

之后每次 push 到 main 分支会自动部署。

```bash
# 推送到 GitHub
git init
git add .
git commit -m "init: face blur tool"
git remote add origin https://github.com/你的用户名/face-blur-web.git
git push -u origin main
```

## 技术栈

- Next.js 14 (Static Export)
- face-api.js (浏览器端人脸检测)
- Canvas API (图像处理)
