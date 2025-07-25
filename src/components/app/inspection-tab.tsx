"use client";

import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import {
  analyzeScrewDefects,
  AnalyzeScrewDefectsOutput,
} from "@/ai/flows/analyze-screw-defects";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { fileToBase64 } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  TestTubeDiagonal,
  CheckCircle,
  XCircle,
  AlertTriangle,
} from "lucide-react";
import Image from "next/image";
import type { InspectionResult } from "./results-tab";

const formSchema = z.object({
  cameraFeed: z
    .custom<FileList>()
    .refine(
      (files) => files && files.length === 1,
      "A single camera image is required."
    )
    .refine(
      (files) => files && files[0].type.startsWith("image/"),
      "Only image files are allowed."
    ),
  sensor3dData: z.string().min(1, "3D sensor data is required."),
  normalAiProfile: z.string().min(1, "A normal AI profile (Model ID) is required."),
});

type InspectionTabProps = {
  modelId: string | null;
  onInspectionComplete: (result: InspectionResult) => void;
};

export function InspectionTab({ modelId, onInspectionComplete }: InspectionTabProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] =
    useState<AnalyzeScrewDefectsOutput | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const { toast } = useToast();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      cameraFeed: undefined,
      sensor3dData: "Sample 3D sensor data: { height: 25.4, thread_depth: 1.5, form_deviation: 0.02 }",
      normalAiProfile: modelId || "",
    },
  });

  useEffect(() => {
    if (modelId) {
      form.setValue("normalAiProfile", modelId);
    }
  }, [modelId, form]);

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    setIsLoading(true);
    setAnalysisResult(null);
    try {
      const cameraFeedDataUri = await fileToBase64(values.cameraFeed[0]);

      const result = await analyzeScrewDefects({
        cameraFeedDataUri,
        sensor3dData: values.sensor3dData,
        normalAiProfile: values.normalAiProfile,
      });

      setAnalysisResult(result);
      onInspectionComplete({
        ...result,
        image: result.defectVisualizationDataUri || cameraFeedDataUri,
        timestamp: new Date().toISOString(),
      });

      toast({
        title: "Analysis Complete",
        description: `Screw status: ${result.screwStatus}`,
      });
    } catch (error) {
      console.error("Error analyzing screw:", error);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: "An error occurred while analyzing the screw.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      form.setValue("cameraFeed", event.target.files as FileList);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  return (
    <div className="grid md:grid-cols-2 gap-8 items-start">
      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl font-headline">
            <TestTubeDiagonal />
            Live Inspection
          </CardTitle>
          <CardDescription>
            Upload a screw image and provide sensor data to analyze it against
            the trained profile.
          </CardDescription>
        </CardHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <CardContent className="space-y-4">
              <FormField
                control={form.control}
                name="cameraFeed"
                render={() => (
                  <FormItem>
                    <FormLabel>Camera Feed Image</FormLabel>
                    <FormControl>
                      <Input
                        type="file"
                        onChange={handleFileChange}
                        className="file:text-foreground"
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="sensor3dData"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>3D Sensor Data</FormLabel>
                    <FormControl>
                      <Textarea placeholder="Enter 3D sensor data..." {...field} rows={4}/>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="normalAiProfile"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Normal AI Profile (Model ID)</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Train a model in Reference tab first"
                        {...field}
                      />
                    </FormControl>
                    {!modelId && (
                      <p className="text-xs text-accent flex items-center gap-1 mt-1">
                        <AlertTriangle size={14} /> Train a model in the
                        Reference tab to get a Model ID.
                      </p>
                    )}
                    <FormMessage />
                  </FormItem>
                )}
              />
            </CardContent>
            <CardFooter>
              <Button type="submit" disabled={isLoading} className="w-full">
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing...
                  </>
                ) : (
                  "Analyze Screw"
                )}
              </Button>
            </CardFooter>
          </form>
        </Form>
      </Card>

      <div className="space-y-4">
        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle className="text-xl font-headline">Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            <Image
              src={
                analysisResult?.defectVisualizationDataUri ||
                previewUrl ||
                "https://placehold.co/600x400.png"
              }
              alt="Screw visualization"
              width={600}
              height={400}
              className="rounded-lg border border-border object-cover w-full aspect-video"
              data-ai-hint="screw industrial"
            />
          </CardContent>
        </Card>
        {analysisResult && (
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="text-xl font-headline">Analysis Result</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center gap-4">
              <p className="font-semibold">Status:</p>
              <Badge
                variant={analysisResult.screwStatus === "OK" ? "default" : "destructive"}
              >
                {analysisResult.screwStatus === "OK" ? (
                  <CheckCircle className="mr-2 h-4 w-4" />
                ) : (
                  <XCircle className="mr-2 h-4 w-4" />
                )}
                {analysisResult.screwStatus}
              </Badge>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
