import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "audiovec — Emotion Embedding",
  description:
    "Upload speech audio to predict emotion and extract a 256-dimensional embedding vector.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
