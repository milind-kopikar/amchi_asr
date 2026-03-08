import type { Metadata } from "next";
import { Noto_Sans_Devanagari, Inter } from "next/font/google";
import "./globals.css";

const devanagari = Noto_Sans_Devanagari({
  variable: "--font-devanagari",
  subsets: ["devanagari"],
  weight: ["400", "500", "700"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Amchi ASR — Deaf Speech Transcriber",
  description:
    "Demo: AI-powered transcription of deaf Marathi speech using fine-tuned IndicConformer + Gemini post-processing",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="mr">
      <body
        className={`${devanagari.variable} ${inter.variable} antialiased bg-gray-50 min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}
