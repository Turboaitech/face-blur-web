export const metadata = {
  title: "人脸模糊工具 | Face Blur Tool",
  description: "在线人脸模糊工具，纯前端处理，保护隐私",
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
