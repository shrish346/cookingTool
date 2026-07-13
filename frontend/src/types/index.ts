// Types matching the backend API (src/schemas.py)

/** Where a piece of the recipe came from. */
export type Provenance =
  | 'video'      // observed in the source video
  | 'reference'  // taken from a published recipe; cites source_id
  | 'model'      // the model's own cooking knowledge; an estimate

export type StepKind =
  | 'gather_tools'
  | 'gather_ingredients'
  | 'prep'
  | 'cook'
  | 'assemble'
  | 'rest'
  | 'serve'
  | 'technique'
  | 'safety'

export interface Source {
  id: string
  title: string
  url?: string
  site?: string
}

export interface Tool {
  id: string
  name: string
  essential: boolean
  substitute?: string
  provenance: Provenance
  source_id?: string
}

export interface Ingredient {
  id: string
  name: string
  quantity: number
  unit: string
  preparation?: string
  optional: boolean
  provenance: Provenance
  source_id?: string
  note?: string
}

/** Side-panel content. Schema only - nothing populates these yet. */
export interface Artifact {
  type: string
  title?: string
  payload: Record<string, unknown>
}

export interface Step {
  id: string
  order: number
  kind: StepKind
  title?: string
  display_title?: string  // Auto-generated if title is missing
  instruction: string
  detail?: string
  doneness_cue?: string
  duration_minutes?: number
  tips?: string[]
  ingredient_ids: string[]
  tool_ids: string[]
  depends_on: string[]
  provenance: Provenance
  source_id?: string
  artifacts: Artifact[]
  /** False for steps the video never showed - they will never get a clip. */
  has_video_clip: boolean
  video_clip_url?: string
  start_timestamp_seconds?: number
  end_timestamp_seconds?: number
  micro_action_ids?: number[]
}

export interface Recipe {
  schema_version: number
  title: string
  description?: string
  ingredients: Ingredient[]
  tools: Tool[]
  sources: Source[]
  steps: Step[]
  servings: number
  difficulty?: string
  prep_time_minutes?: number
  cook_time_minutes?: number
  total_time_minutes?: number
  source_url?: string
  cuisine?: string
  tags?: string[]
  video_id?: string
  video_fps?: number
  clips_ready: boolean
  expansion_failed?: boolean
}

export type VideoSource = 'youtube' | 'tiktok' | 'instagram' | 'unknown'

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

