"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { FileDown, FileStack, CheckCircle, XCircle } from "lucide-react";
import Image from "next/image";
import { format } from "date-fns";
import type { AnalyzeScrewDefectsOutput } from "@/ai/flows/analyze-screw-defects";

export type InspectionResult = AnalyzeScrewDefectsOutput & {
  image: string;
  timestamp: string;
};

type ResultsTabProps = {
  results: InspectionResult[];
};

export function ResultsTab({ results }: ResultsTabProps) {
  const exportResults = () => {
    const csvContent =
      "data:text/csv;charset=utf-8," +
      "Timestamp,Status,DefectDetected,ImageURL\n" +
      results
        .map(
          (r) =>
            `${r.timestamp},${r.screwStatus},${
              r.defectDetected
            },${r.image.substring(0, 50)}...`
        )
        .join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "inspection_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Card className="max-w-6xl mx-auto shadow-lg">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2 text-xl font-headline">
            <FileStack />
            Inspection History
          </CardTitle>
          <CardDescription>
            A log of all past screw inspections. You can export the results as a
            CSV file.
          </CardDescription>
        </div>
        <Button onClick={exportResults} disabled={results.length === 0}>
          <FileDown className="mr-2 h-4 w-4" />
          Export Results
        </Button>
      </CardHeader>
      <CardContent>
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[100px]">Image</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {results.length > 0 ? (
                results.map((result, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Image
                        src={result.image}
                        alt={`Inspection ${index + 1}`}
                        width={64}
                        height={64}
                        className="rounded-md object-cover"
                        data-ai-hint="screw part"
                      />
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          result.screwStatus === "OK" ? "default" : "destructive"
                        }
                      >
                        {result.screwStatus === "OK" ? (
                          <CheckCircle className="mr-2 h-4 w-4" />
                        ) : (
                          <XCircle className="mr-2 h-4 w-4" />
                        )}
                        {result.screwStatus}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {format(new Date(result.timestamp), "yyyy-MM-dd HH:mm:ss")}
                    </TableCell>
                    <TableCell className="font-code text-xs">
                      {result.defectDetected
                        ? "Defect Detected"
                        : "No Defect"}
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={4} className="h-24 text-center">
                    No results yet. Perform an inspection to see the history
                    here.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
