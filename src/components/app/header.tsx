import { ScanSearch } from "lucide-react";

export function AppHeader() {
  return (
    <header className="p-4 border-b border-border shadow-sm">
      <div className="container mx-auto flex items-center gap-3">
        <ScanSearch className="h-7 w-7 text-primary" />
        <h1 className="text-2xl font-headline font-bold text-foreground">
          InspectionAI
        </h1>
      </div>
    </header>
  );
}
