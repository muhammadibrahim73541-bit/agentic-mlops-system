import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Use the forwarded backend URL in Codespaces
export const API_BASE = "https://didactic-giggle-x5q77xwprg5v26xjv-8000.app.github.dev"
