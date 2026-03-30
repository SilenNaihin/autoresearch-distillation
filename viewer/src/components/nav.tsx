"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/scenarios", label: "Scenarios" },
  { href: "/compare", label: "Compare" },
  { href: "/training", label: "Training" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 z-50 w-full border-b border-border bg-bg/80 backdrop-blur-sm">
      <div className="mx-auto flex h-12 max-w-[1200px] items-center gap-6 px-6">
        <Link
          href="/"
          className="mr-2 text-sm font-semibold tracking-tight text-text"
        >
          GAIA
        </Link>
        <div className="flex items-center gap-1">
          {links.map((link) => {
            const isActive =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-md px-3 py-1.5 text-[13px] transition-colors ${
                  isActive
                    ? "text-text"
                    : "text-text-tertiary hover:text-text-secondary"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
