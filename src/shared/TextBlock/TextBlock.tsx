import { FC } from "react"
import { createUseStyles } from "react-jss"

const useStyles = createUseStyles({
    textBlock: {
        backgroundColor: "#1E1E1E",
        width: "80%",
        marginLeft: "10%",
        padding: "7px",
        marginTop: "7px",
        border: "1px solid #ffffff",
        borderRadius: "4px",
        display: "flex",
        flexDirection: "column",
        textWrap: "wrap"
    },
});

type Props = {
    text: string
}

export const TextBlock: FC<Props> = ({text}) => {
    const classes = useStyles()

    return (
        <div className={classes.textBlock}>{text}</div>
    )
}

export default TextBlock