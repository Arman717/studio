"use client";

import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { generateDefectProfile } from "@/ai/flows/generate-defect-profile";
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
import { useToast } from "@/hooks/use-toast";
import { fileToBase64 } from "@/lib/utils";
import { Bot, CheckCircle, Loader2, Upload } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import Image from "next/image";

const formSchema = z.object({
  referenceImages: z
    .custom<FileList>()
    .refine(
      (files) => files && files.length > 0,
      "At least one reference image is required."
    )
    .refine(
      (files) =>
        Array.from(files).every((file) => file.type.startsWith("image/")),
      "Only image files are allowed."
    ),
});

type ReferenceTabProps = {
  onModelTrained: (modelId: string) => void;
};

export function ReferenceTab({ onModelTrained }: ReferenceTabProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [trainedModelId, setTrainedModelId] = useState<string | null>(null);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const { toast } = useToast();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      referenceImages: undefined,
    },
  });

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    setIsLoading(true);
    setProgress(0);
    setTrainedModelId(null);
    try {
      const imageFiles = Array.from(values.referenceImages);
      const base64Images = await Promise.all(
        imageFiles.map((file) => fileToBase64(file))
      );

      const result = await generateDefectProfile({
        referenceImages: base64Images,
      });

      onModelTrained(result.modelId);
      setTrainedModelId(result.modelId);

      toast({
        title: "Model Trained Successfully",
        description: `New defect profile created with ID: ${result.modelId}`,
      });
    } catch (error) {
      console.error("Error training model:", error);
      toast({
        variant: "destructive",
        title: "Training Failed",
        description: "An error occurred while training the AI model.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    let timer: NodeJS.Timer | undefined;
    if (isLoading) {
      timer = setInterval(() => {
        setProgress(prev => (prev >= 95 ? 95 : prev + 1));
      }, 200);
    } else {
      setProgress(100);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isLoading]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      form.setValue("referenceImages", files);
      const urls = Array.from(files).map((file) => URL.createObjectURL(file));
      setPreviewUrls(urls);
    }
  };

  return (
    <Card className="max-w-3xl mx-auto shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl font-headline">
          <Bot />
          Reference Mode: Train Defect Profile
        </CardTitle>
        <CardDescription>
          Upload one or more images of a defect-free screw to train the AI and
          create a "normal" profile for comparison.
        </CardDescription>
      </CardHeader>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)}>
          <CardContent className="space-y-6">
            <FormField
              control={form.control}
              name="referenceImages"
              render={() => (
                <FormItem>
                  <FormLabel>Reference Screw Images</FormLabel>
                  <FormControl>
                    <div className="flex items-center justify-center w-full">
                      <label
                        htmlFor="dropzone-file"
                        className="flex flex-col items-center justify-center w-full h-64 border-2 border-border border-dashed rounded-lg cursor-pointer bg-card hover:bg-secondary/50"
                      >
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          <Upload className="w-8 h-8 mb-4 text-muted-foreground" />
                          <p className="mb-2 text-sm text-muted-foreground">
                            <span className="font-semibold text-primary">Click to upload</span> or drag
                            and drop
                          </p>
                          <p className="text-xs text-muted-foreground">
                            PNG, JPG, or other image formats
                          </p>
                        </div>
                        <Input
                          id="dropzone-file"
                          type="file"
                          className="hidden"
                          multiple
                          onChange={handleFileChange}
                        />
                      </label>
                    </div>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            {previewUrls.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Image Previews:</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {previewUrls.map((url, index) => (
                    <div key={index} className="relative aspect-square">
                        <Image
                            src={url}
                            alt={`Preview ${index + 1}`}
                            fill
                            className="rounded-md object-cover"
                        />
                    </div>
                  ))}
                </div>
              </div>
            )}
            {trainedModelId && (
              <div className="p-4 rounded-md bg-primary/10 border border-primary/20 flex items-center gap-3">
                <CheckCircle className="text-primary h-6 w-6" />
                <div>
                  <p className="font-semibold text-primary">
                    Training Complete!
                  </p>
                  <p className="text-sm text-primary/80 font-code">
                    Model ID: {trainedModelId}
                  </p>
                </div>
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button type="submit" disabled={isLoading} className="w-full">
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Training Model...
                </>
              ) : (
                "Train Model"
              )}
            </Button>
            {isLoading && (
              <Progress value={progress} className="w-full mt-4" />
            )}
          </CardFooter>
        </form>
      </Form>
    </Card>
  );
}
