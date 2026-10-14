import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "agentproof | Agent Reliability Lab",
  description: "Reliability and failure analysis dashboard for ChemDesk agent",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
