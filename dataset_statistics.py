# dataset statistics WP
import sys, os
from data.util import *
from rake_nltk import Rake
from tqdm import trange

data_folder = 'data/writingPrompts/'
data = ["train", "test", "valid"]

num_stories = 0.0
prompt_len = 0.0
story_len = 0.0
story_pars = 0.0
phrase_num = 0.0
phrase_len = 0.0

for name in data:
    with open(data_folder + name + ".wp_source") as fs:
        with open(data_folder + name + ".wp_target") as ft:
            stories = ft.readlines()
            prompt = fs.readlines()
            prompts = list(zip(prompt, stories))
    num_stories += len(prompts)

    for i in trange(len(prompts)):
        prompt, story = prompts[i]
        prompt = prompt.strip()
        story = story.strip()

        prompt = re.sub('\[ (.*) \]', '', prompt)
        prompt_len += len(prompt.split())

        story_len += len(story.split())

        #pp = get_paragraph(story)
        pp = story.split('<newline><newline>')
        pp = [p.replace('\n', '').strip() for p in pp]
        r = Rake(min_length=1, max_length=4)
        keys = [extract_keywords(text, r) for text in pp]

        story_pars += len(pp)
        phrase_num += sum([len(k) for k in keys])
        phrase_len += sum([sum([len(k.split()) for k in kl]) for kl in keys])


print(num_stories)
print(prompt_len / num_stories)
print(story_len / num_stories)
print(story_pars / num_stories)
print(phrase_num / story_pars)
print(phrase_len / phrase_num)


# dataset statistics WI
import sys, os
from data.util import *
from rake_nltk import Rake
from tqdm import trange

data_folder = 'data/wikiPlots/'

num_stories = 0.0
prompt_len = 0.0
story_len = 0.0
story_pars = 0.0
phrase_num = 0.0
phrase_len = 0.0

with open(data_folder + "titles") as fs:
    with open(data_folder + "plots_paragraph") as ft:
        plots = ft.readlines()
        titles = fs.readlines()

prompts = [(t, p) for t, p in zip(titles, plots) if t.strip() != '' and p.strip() != '']
num_stories += len(prompts)

for i in trange(len(prompts)):
    prompt, story = prompts[i]
    prompt = prompt.strip()
    story = story.strip()

    prompt_len += len(prompt.split())

    story_len += len(story.split())

    # pp = get_paragraph(story)
    pp = story.split('<newline><newline>')
    pp = [p.replace('\n', '').strip() for p in pp]
    r = Rake(min_length=1, max_length=4)
    keys = [extract_keywords(text, r) for text in pp]

    story_pars += len(pp)
    phrase_num += sum([len(k) for k in keys])
    phrase_len += sum([sum([len(k.split()) for k in kl]) for kl in keys])

print(num_stories)
print(prompt_len / num_stories)
print(story_len / num_stories)
print(story_pars / num_stories)
print(phrase_num / story_pars)
print(phrase_len / phrase_num)
