export const metadata = {
  title: "Face Blur & Isolate Tool",
  description: "Auto-detect faces: blur them or isolate on white background. 100% browser-side.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
