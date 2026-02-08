import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Strip FN_CALL=... from chat text so it is never shown in the UI. */
export function stripFnCallFromDisplay(text: string): string {
  if (!text || typeof text !== "string") return text;
  return text.replace(/\s*FN_CALL=(?:True|False)\s*/gi, " ").trim();
}
