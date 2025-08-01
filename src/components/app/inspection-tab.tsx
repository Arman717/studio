"use client";

import Webcam from "react-webcam";
import { useState, useEffect, useRef, type MouseEvent } from "react";
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
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  TestTubeDiagonal,
  CheckCircle,
  XCircle,
  AlertTriangle,
} from "lucide-react";
import NextImage from "next/image";
import type { InspectionResult } from "./results-tab";

const formSchema = z.object({
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
  const webcamRef = useRef<Webcam>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [showCamera, setShowCamera] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const [cropRect, setCropRect] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
  const [clipStyle, setClipStyle] = useState<React.CSSProperties | undefined>();
  const { toast } = useToast();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      sensor3dData:
        "Sample 3D sensor data: { height: 25.4, thread_depth: 1.5, form_deviation: 0.02 }",
      normalAiProfile: modelId || "",
    },
  });

  useEffect(() => {
    setShowCamera(true);
  }, []);

  useEffect(() => {
    if (modelId) {
      form.setValue("normalAiProfile", modelId);
    }
  }, [modelId, form]);

  const beginSelect = (e: MouseEvent<HTMLDivElement>) => {
    if (isLoading) return;
    const rect = overlayRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCropRect({ x, y, width: 0, height: 0 });
    setSelecting(true);
  };

  const updateSelect = (e: MouseEvent<HTMLDivElement>) => {
    if (!selecting || !cropRect) return;
    const rect = overlayRef.current!.getBoundingClientRect();
    const width = e.clientX - rect.left - cropRect.x;
    const height = e.clientY - rect.top - cropRect.y;
    setCropRect({ ...cropRect, width, height });
  };

  const endSelect = () => {
    setSelecting(false);
  };

  useEffect(() => {
    if (!cropRect || !overlayRef.current) {
      setClipStyle(undefined);
      return;
    }
    const rect = overlayRef.current.getBoundingClientRect();
    const left = Math.min(cropRect.x, cropRect.x + cropRect.width);
    const top = Math.min(cropRect.y, cropRect.y + cropRect.height);
    const width = Math.abs(cropRect.width);
    const height = Math.abs(cropRect.height);
    const right = rect.width - (left + width);
    const bottom = rect.height - (top + height);
    setClipStyle({ clipPath: `inset(${top}px ${right}px ${bottom}px ${left}px)` });
  }, [cropRect]);

  async function cropImage(dataUri: string): Promise<string> {
    if (!cropRect) return dataUri;
    const img = new Image();
    img.src = dataUri;
    await new Promise<void>((res) => {
      img.onload = () => res();
    });
    const canvas = document.createElement("canvas");
    const videoW = img.width;
    const videoH = img.height;
    const overlay = overlayRef.current?.getBoundingClientRect();
    const scaleX = overlay ? videoW / overlay.width : 1;
    const scaleY = overlay ? videoH / overlay.height : 1;
    const x = Math.min(cropRect.x, cropRect.x + cropRect.width) * scaleX;
    const y = Math.min(cropRect.y, cropRect.y + cropRect.height) * scaleY;
    const w = Math.abs(cropRect.width) * scaleX;
    const h = Math.abs(cropRect.height) * scaleY;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, x, y, w, h, 0, 0, w, h);
    return canvas.toDataURL("image/png");
  }

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    setIsLoading(true);
    setAnalysisResult(null);
    try {
      const screenshot = webcamRef.current?.getScreenshot();
      if (!screenshot) {
        toast({
          variant: "destructive",
          title: "Camera Error",
          description: "Unable to capture image from webcam.",
        });
        setIsLoading(false);
        return;
      }

      const cameraFeedDataUri = await cropImage(screenshot);
      setPreviewUrl(cameraFeedDataUri);

      const result = await analyzeScrewDefects({
        cameraFeedDataUri,
        sensor3dData: values.sensor3dData,
        normalAiProfile: values.normalAiProfile,
        modelId: values.normalAiProfile,
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


  return (
    <div className="grid md:grid-cols-2 gap-8 items-start">
      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl font-headline">
            <TestTubeDiagonal />
            Live Inspection
          </CardTitle>
          <CardDescription>
            Select an area of interest on the camera feed and provide sensor
            data to analyze it against the trained profile.
          </CardDescription>
        </CardHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <CardContent className="space-y-4">
              {showCamera && (
                <div className="relative">
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    className="w-full rounded-md"
                    style={!selecting && clipStyle ? clipStyle : undefined}
                  />
                  <div
                    ref={overlayRef}
                    className={`absolute inset-0 ${selecting ? "cursor-crosshair" : ""}`}
                    onMouseDown={beginSelect}
                    onMouseMove={updateSelect}
                    onMouseUp={endSelect}
                    onMouseLeave={endSelect}
                  >
                    {cropRect && (
                      <>
                        <div
                          className="absolute border border-red-500"
                          style={{
                            left: Math.min(cropRect.x, cropRect.x + cropRect.width),
                            top: Math.min(cropRect.y, cropRect.y + cropRect.height),
                            width: Math.abs(cropRect.width),
                            height: Math.abs(cropRect.height),
                          }}
                        />
                        {!selecting && (
                          <>
                            <div
                              className="absolute bg-black/80"
                              style={{
                                top: 0,
                                left: 0,
                                right: 0,
                                height: Math.min(cropRect.y, cropRect.y + cropRect.height),
                              }}
                            />
                            <div
                              className="absolute bg-black/80"
                              style={{
                                top:
                                  Math.min(cropRect.y, cropRect.y + cropRect.height) +
                                  Math.abs(cropRect.height),
                                left: 0,
                                right: 0,
                                bottom: 0,
                              }}
                            />
                            <div
                              className="absolute bg-black/80"
                              style={{
                                top: Math.min(cropRect.y, cropRect.y + cropRect.height),
                                left: 0,
                                width: Math.min(cropRect.x, cropRect.x + cropRect.width),
                                height: Math.abs(cropRect.height),
                              }}
                            />
                            <div
                              className="absolute bg-black/80"
                              style={{
                                top: Math.min(cropRect.y, cropRect.y + cropRect.height),
                                left:
                                  Math.min(cropRect.x, cropRect.x + cropRect.width) +
                                  Math.abs(cropRect.width),
                                right: 0,
                                height: Math.abs(cropRect.height),
                              }}
                            />
                          </>
                        )}
                      </>
                    )}
                  </div>
                </div>
              )}
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
            <NextImage
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
