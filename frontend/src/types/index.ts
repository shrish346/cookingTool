// Types matching the backend API

export interface Ingredient {
  name: string
  quantity: number
  unit: string
  preparation?: string
}

export interface Step {
  order: number
  title?: string
  display_title?: string  // Auto-generated if title is missing
  instruction: string
  duration_minutes?: number
  tips?: string[]
  start_frame_index?: number
  end_frame_index?: number
  video_clip_url?: string
}

export interface Recipe {
  title: string
  description?: string
  ingredients: Ingredient[]
  steps: Step[]
  servings: number
  prep_time_minutes?: number
  cook_time_minutes?: number
  source_url?: string
  cusine?: string
  tags?: string[]
  video_id?: string
  video_fps?: number
  clips_ready: boolean
}

export type VideoSource = 'youtube' | 'tiktok' | 'unknown'

export interface ValidateResponse {
  valid: boolean
  source: VideoSource
  video_id?: string
  error?: string
}

export type JobStatus = 
  | 'pending'
  | 'downloading'
  | 'analyzing'
  | 'generating'
  | 'extracting_clips'
  | 'uploading'
  | 'completed'
  | 'failed'

export interface ProcessResponse {
  job_id: string
  video_id: string
  status: string
  message: string
}

export interface StatusResponse {
  job_id: string
  status: JobStatus
  progress: number
  message: string
  recipe?: Recipe
  clips_ready: boolean
  error?: string
}

