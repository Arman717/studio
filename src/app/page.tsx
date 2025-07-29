"use client";

import { useState } from "react";
import { AppHeader } from "@/components/app/header";
import { ReferenceTab } from "@/components/app/reference-tab";
import { InspectionTab } from "@/components/app/inspection-tab";
import { ResultsTab, type InspectionResult } from "@/components/app/results-tab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Bot, FileStack, TestTubeDiagonal, Settings } from "lucide-react";
import { ControlTab } from "@/components/app/control-tab";

export default function Home() {
  const [modelId, setModelId] = useState<string | null>(null);
  const [results, setResults] = useState<InspectionResult[]>([]);

  const addResult = (result: InspectionResult) => {
    setResults((prev) => [result, ...prev]);
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <AppHeader />
      <main className="flex-grow container mx-auto p-4 md:p-8">
        <Tabs defaultValue="reference" className="w-full">
          <TabsList className="grid w-full grid-cols-4 max-w-3xl mx-auto">
            <TabsTrigger value="reference">
              <Bot className="mr-2 h-4 w-4" /> Reference
            </TabsTrigger>
            <TabsTrigger value="inspection">
              <TestTubeDiagonal className="mr-2 h-4 w-4" /> Inspection
            </TabsTrigger>
            <TabsTrigger value="results">
              <FileStack className="mr-2 h-4 w-4" /> Results
            </TabsTrigger>
            <TabsTrigger value="control">
              <Settings className="mr-2 h-4 w-4" /> Control
            </TabsTrigger>
          </TabsList>
          <TabsContent value="reference" className="mt-6">
            <ReferenceTab onModelTrained={setModelId} />
          </TabsContent>
          <TabsContent value="inspection" className="mt-6">
            <InspectionTab modelId={modelId} onInspectionComplete={addResult} />
          </TabsContent>
          <TabsContent value="results" className="mt-6">
            <ResultsTab results={results} />
          </TabsContent>
          <TabsContent value="control" className="mt-6">
            <ControlTab />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
