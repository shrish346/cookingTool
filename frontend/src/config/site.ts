/**
 * Notice mode — the site's one manual switch.
 *
 * A non-empty `SITE_NOTICE` replaces the link-entry landing with the notice plus a
 * list of recipes that already exist in object storage. Set it back to `''` and the
 * normal app comes straight back; nothing else needs touching.
 *
 * Nothing in this mode talks to the backend. Recipes and their clips are read
 * directly from the clips Worker (see api/storedRecipe.ts), so the site keeps
 * working with the API, Redis and the download worker all switched off.
 */

export const SITE_NOTICE =
  "We are currently exploring new hosting options for our video link to recipe " +
  "pipeline. In the interim, here are recipes you can try. We're also working on " +
  'exciting new features, all coming soon!'

export interface BrowseRecipe {
  /** Object-storage prefix — the video id, namespaced (`tt-`/`ig-`) for non-YouTube. */
  id: string
  /** Mirrors the `title` inside that recipe's stored recipe.json. */
  title: string
}

/**
 * Curated: every entry here has a `{id}/recipe.json` in the bucket with clips
 * rendered. Regenerate with `python3 scripts/list_r2_recipes.py` after processing
 * new videos, then drop the junk/test entries it prints.
 */
export const BROWSE_RECIPES: ReadonlyArray<BrowseRecipe> = [
  { id: '2KyDrqiiZdU', title: 'High-Protein Alfredo Pasta' },
  { id: 'xRAcjC2ifAM', title: '20-Minute Butter Chicken' },
  { id: '6Au_WwSLJ3Q', title: 'Ultimate Chicken Biryani' },
  { id: '8NieSQurtxk', title: 'Pad See Ew' },
  { id: 'uKEYtSlnqMI', title: 'Sheet Pan Garlic Parmesan Chicken and Potatoes' },
  { id: 'unv5G420hqY', title: 'Kimchi-jjigae (Kimchi Stew)' },
  { id: 'hvZs62R3VQg', title: 'Creamy Tomato & Mozzarella Pasta' },
  { id: 'lEOlVnM9gD0', title: 'One-Pan Greek Chicken Rice Bowls' },
  { id: '66zdFWr4C9c', title: 'Mushroom and Egg Quesadilla' },
  { id: 'Bry3_AvtoyE', title: 'Empty Salna (South Indian Gravy)' },
  { id: 'tt-ZP8tYb9TJ', title: 'Argentinian Steak Frites with Chimichurri' },
]

export const noticeMode = SITE_NOTICE.trim().length > 0
